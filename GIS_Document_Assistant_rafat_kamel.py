__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import json
import tempfile
from datetime import datetime
from collections import Counter
import pandas as pd
import streamlit as st

# App config
st.set_page_config(
    page_title="GIS Document Assistant",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI/UX
st.markdown("""
<style>
    [data-testid="stChatMessage"] { 
        border-radius: 12px; 
        margin-bottom: 12px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #0078d4;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 13px;
        color: #333;
    }
    .stat-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        color: #0078d4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .stat-label { 
        font-size: 13px; 
        color: #666; 
        font-weight: 500; 
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .footer {
        font-size: 12px;
        color: #888;
        text-align: center;
        margin-top: 40px;
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

        pages = PyPDFLoader(pdf_path).load()
        for p in pages:
            p.metadata["pdf_name"] = pdf_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(pages)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks, embedding=self.embeddings,
                collection_name="gis_streamlit"
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
            key = f"{c.metadata.get('pdf_name','?')} · p.{c.metadata.get('page','?')}"
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
                    "text": c.page_content[:200] + "...",
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


# Init session state
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar layout
with st.sidebar:
    st.title("🗺️ Geo-AI Assistant")
    st.caption("Empowering GIS with LLMs")
    st.divider()

    st.subheader("🔑 API Configuration")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter AIza...")

    if api_key and st.session_state.assistant is None:
        st.session_state.assistant = GISDocumentAssistant(api_key)

    if not api_key:
        st.warning("⚠️ Please provide a valid Gemini API key.")
        st.stop()

    st.divider()

    st.subheader("📄 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload GIS PDF documents",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        assistant: GISDocumentAssistant = st.session_state.assistant
        already_loaded = set(assistant.loaded_pdfs)

        for uploaded in uploaded_files:
            if uploaded.name not in already_loaded:
                with st.spinner(f"Ingesting {uploaded.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    n = assistant.load_pdf(tmp_path, uploaded.name)
                    os.unlink(tmp_path)
                st.toast(f"Loaded: {uploaded.name} ({n} chunks)", icon="✅")

    if st.session_state.assistant and st.session_state.assistant.loaded_pdfs:
        with st.expander("📁 Indexed Documents", expanded=True):
            for pdf in st.session_state.assistant.loaded_pdfs:
                st.markdown(f"- `{pdf}`")

    st.divider()
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk size", 200, 1000, 500, 50)
        
    if st.session_state.assistant and st.session_state.assistant.chat_history:
        json_str = st.session_state.assistant.export_json()
        fname = f"gis_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button("⬇️ Export History", data=json_str, file_name=fname, mime="application/json", use_container_width=True)

    if st.button("🗑️ Reset Session", use_container_width=True, type="primary"):
        st.session_state.messages = []
        if st.session_state.assistant:
            st.session_state.assistant.chat_history = []
        st.rerun()

    st.markdown('<div class="footer">Developed for GIS Professionals</div>', unsafe_allow_html=True)


# Main UI
assistant: GISDocumentAssistant = st.session_state.assistant

col1, col2 = st.columns([3, 2])
with col1:
    st.header("Document Intelligence")
    st.caption("Ask queries in English or Arabic regarding your spatial data & docs.")

if assistant and assistant.loaded_pdfs:
    with col2:
        c1, c2, c3 = st.columns(3)
        total_q = len(assistant.chat_history)
        with c1:
            st.markdown(f'<div class="stat-box">{len(assistant.loaded_pdfs)}<br><span class="stat-label">Docs</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box">{assistant.total_chunks}<br><span class="stat-label">Chunks</span></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box">{total_q}<br><span class="stat-label">Queries</span></div>', unsafe_allow_html=True)

st.divider()

if not assistant or not assistant.loaded_pdfs:
    st.info("👋 Welcome! Please upload your GIS PDF documents from the sidebar to begin.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("🔍 View References", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>[{i}] {src["pdf"]} (Page {src["page"]})</b><br>{src["text"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if question := st.chat_input("Query your GIS documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving spatial context..."):
            entry = assistant.ask(question)

        flag = "🇸🇦" if entry["language"] == "ar" else "🇬🇧"
        response_text = f"{flag} {entry['answer']}"
        st.markdown(response_text)

        with st.expander("🔍 View References", expanded=False):
            for i, src in enumerate(entry["sources"], 1):
                st.markdown(
                    f'<div class="source-card">'
                    f'<b>[{i}] {src["pdf"]} (Page {src["page"]})</b><br>{src["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": entry["sources"],
    })

# Analytics section
if assistant.chat_history and assistant.page_usage:
    st.divider()
    with st.expander("📊 Document Analytics", expanded=False):
        top_pages = assistant.page_usage.most_common(10)
        df = pd.DataFrame(top_pages, columns=["Reference", "Hits"])
        st.bar_chart(df.set_index("Reference"))