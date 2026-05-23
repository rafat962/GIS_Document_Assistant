import os
import json
import tempfile
from datetime import datetime
from collections import Counter
import pandas as pd
import streamlit as st

# Force fully disabled telemetry globally before any chromadb interactions
os.environ["ANONYMIZED_TELEMETRY"] = "False"

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Page Configuration
st.set_page_config(
    page_title="GIS Document Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Dark UI Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stChatMessage"] { 
        border-radius: 16px; 
        margin-bottom: 14px;
        padding: 20px;
        border: 1px solid #30363d;
        background-color: #161b22;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .source-card {
        background: #0d1117;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 13.5px;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    .stat-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        color: #58a6ff;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }
    .stat-label { 
        font-size: 12px; 
        color: #8b949e; 
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #c9d1d9 !important;
    }
    .footer {
        font-size: 11px;
        color: #484f58;
        text-align: center;
        margin-top: 50px;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


class GISDocumentAssistant:
    def __init__(self, api_key: str):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        os.environ["GOOGLE_API_KEY"] = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        self.vectorstore = None
        self.loaded_pdfs: list[str] = []
        self.total_chunks: int = 0
        self.chat_history: list[dict] = []
        self.page_usage: Counter = Counter()

    def load_pdf(self, pdf_path: str, pdf_name: str, chunk_size=500, chunk_overlap=50) -> int:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        import chromadb
        from chromadb.config import Settings

        pages = PyPDFLoader(pdf_path).load()
        for p in pages:
            p.metadata["pdf_name"] = pdf_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(pages)

        if self.vectorstore is None:
            isolated_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                client=isolated_client,
                collection_name="gis_isolated_collection"
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.loaded_pdfs.append(pdf_name)
        self.total_chunks += len(chunks)
        return len(chunks)

    def _detect_language(self, text: str) -> str:
        arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        return "ar" if arabic > len(text) * 0.2 else "en"

    def _build_prompt(self, question: str, context: str, lang: str) -> str:
        if lang == "ar":
            return (
                "أنت مساعد متخصص في نظم المعلومات الجغرافية (GIS).\n"
                "استخدم السياق التالي للإجابة باللغة العربية.\n"
                "إذا لم تجد الإجابة قل: لا تتوفر معلومات كافية في الوثائق المتاحة.\n\n"
                f"السياق:\n{context}\n\nالسؤال: {question}\n\nالإجابة:"
            )
        return (
            "You are a GIS specialist assistant.\n"
            "Use the context below to answer. If the answer is not in the context, "
            "say: There is not enough information in the available documents.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    def ask(self, question: str, k: int = 3) -> dict:
        lang = self._detect_language(question)
        chunks = self.vectorstore.similarity_search(question, k=k)
        context = "\n\n".join(c.page_content for c in chunks)
        answer = self.llm.invoke(self._build_prompt(question, context, lang)).content

        for c in chunks:
            key = f"{c.metadata.get('pdf_name','?')} (p. {c.metadata.get('page','?')})"
            self.page_usage[key] += 1

        entry = {
            "id": len(self.chat_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "language": lang,
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "pdf": c.metadata.get("pdf_name", "?"),
                    "page": c.metadata.get("page", "?"),
                    "text": c.page_content[:250] + "...",
                }
                for c in chunks
            ],
        }
        self.chat_history.append(entry)
        return entry

    def export_json(self) -> str:
        payload = {
            "exported_at": datetime.now().isoformat(),
            "pdfs_loaded": self.loaded_pdfs,
            "total_chunks": self.total_chunks,
            "total_questions": len(self.chat_history),
            "page_usage": dict(self.page_usage),
            "conversation": self.chat_history,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


# Session State Management
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = []

assistant: GISDocumentAssistant = st.session_state.assistant

# Sidebar Panel Architecture
with st.sidebar:
    st.title("🌐 GIS Insight")
    st.caption("Advanced Document Intelligence")
    st.divider()

    st.subheader("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste AIzaSy...")

    if api_key and st.session_state.assistant is None:
        st.session_state.assistant = GISDocumentAssistant(api_key)
        st.rerun()

    if not api_key:
        st.warning("🔒 Provide an API key to initialize the engine.")
        st.stop()

    st.divider()

    st.subheader("📂 Geospatial Repository")
    uploaded_files = st.file_uploader(
        "Upload GIS PDF documents",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and assistant:
        already_loaded = set(assistant.loaded_pdfs)
        for uploaded in uploaded_files:
            if uploaded.name not in already_loaded:
                with st.spinner(f"Processing {uploaded.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    n = assistant.load_pdf(tmp_path, uploaded.name)
                    os.unlink(tmp_path)
                st.toast(f"Indexed: {uploaded.name} ({n} chunks)", icon="⚡")
                st.rerun()

    if assistant and assistant.loaded_pdfs:
        with st.expander("📚 Active Core Database", expanded=True):
            for pdf in assistant.loaded_pdfs:
                st.markdown(f"📄 `{pdf}`")

    st.divider()
    
    with st.expander("⚙️ Tuning & Parameters"):
        chunk_size = st.slider("Chunk size", 200, 1000, 500, 50)
        
    if assistant and assistant.chat_history:
        st.subheader("💾 Actions")
        json_str = assistant.export_json()
        fname = f"gis_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button("⬇️ Download Analytics", data=json_str, file_name=fname, mime="application/json", use_container_width=True)

    if st.button("🗑️ Clear Workspace", use_container_width=True, type="primary"):
        st.session_state.messages = []
        if st.session_state.assistant:
            st.session_state.assistant.chat_history = []
            st.session_state.assistant.page_usage = Counter()
        st.rerun()

    st.markdown('<div class="footer">GIS_Document_Assistant_rafat_kamel • v2.0</div>', unsafe_allow_html=True)


# Main Interface Architecture
col_header, col_metrics = st.columns([3, 2])
with col_header:
    st.title("🌐 GIS Document Assistant")
    st.caption("Semantic Neural Search over Geospatial Mapping Documentation Engine.")

# Top Metrics Panel (Real-time Statistics)
if assistant and assistant.loaded_pdfs:
    with col_metrics:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-box">{len(assistant.loaded_pdfs)}<br><span class="stat-label">Indices</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box">{assistant.total_chunks}<br><span class="stat-label">Vectors</span></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box">{len(assistant.chat_history)}<br><span class="stat-label">Queries</span></div>', unsafe_allow_html=True)

st.divider()

if not assistant or not assistant.loaded_pdfs:
    st.info("💡 Get started by uploading specialized GIS map specs or project guidelines via the sidebar.")
    st.stop()

# Chat Stream Execution
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("🔍 Verified Spatial References", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>[{i}] {src["pdf"]} — (Page {src["page"]})</b><br>{src["text"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Input Prompt Handling
if question := st.chat_input("Ask a technical geospatial query..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Executing Vector Similarity Search..."):
            entry = assistant.ask(question)

        flag = "🇸🇦" if entry["language"] == "ar" else "🇬🇧"
        response_content = f"{flag} {entry['answer']}"
        st.markdown(response_content)

        with st.expander("🔍 Verified Spatial References", expanded=False):
            for i, src in enumerate(entry["sources"], 1):
                st.markdown(
                    f'<div class="source-card">'
                    f'<b>[{i}] {src["pdf"]} — (Page {src["page"]})</b><br>{src["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "sources": entry["sources"],
    })
    st.rerun()

# --- Analytics Dashboard & Citation Heatmap Section ---
if assistant and assistant.chat_history and assistant.page_usage:
    st.divider()
    st.subheader("📊 Document Analytics & Citation Heatmap")
    
    top_pages = assistant.page_usage.most_common(10)
    df_stats = pd.DataFrame(top_pages, columns=["Document Reference", "Citation Density"])
    st.bar_chart(df_stats.set_index("Document Reference"))