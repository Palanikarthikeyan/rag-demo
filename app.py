"""
Course Enquiry RAG Chatbot — Production / Streamlit Cloud version
====================================================================
"""
import os

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # fallback for older environments
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================================
# ADMIN CONFIG — edit these, then redeploy. End users never see this section.
# Keep KNOWLEDGE_URLS / KNOWLEDGE_FOLDER in sync with build_index.py.
# ============================================================================
VENDOR_NAME = "Timmins Training Consulting"
KNOWLEDGE_URLS = [
    "https://timmins-consulting.com/",
    "https://timmins-consulting.com/about-us/",
    "https://timmins-consulting.com/our-approach/",
    "https://timmins-consulting.com/our-solution/",
    "https://timmins-consulting.com/domain/embedded-lnux/",
    "https://timmins-consulting.com/training-calendar/public-classes",
    "https://timmins-consulting.com/case-study/",
    "https://timmins-consulting.com/contact-us/",
]
GROQ_MODEL = "qwen/qwen3.6-27b"
KNOWLEDGE_FOLDER = "knowledge"      # extra .pdf/.txt files committed to the repo
FAISS_INDEX_DIR = "faiss_index"     # pre-built index folder from build_index.py
# ============================================================================

SYSTEM_TEMPLATE = """You are the course-enquiry assistant for {vendor_name}, a corporate \
training provider. Prospective students, engineers, and L&D managers ask you about \
courses, formats, prerequisites, certification, HRDC claimability, industries served, \
and how to get in touch.
Rules:
- Answer ONLY using the retrieved context below. Do not invent course names, dates, \
prices, or claims that aren't in the context.
- If the context doesn't cover the question, say so plainly and suggest the person \
contact {vendor_name} directly for that detail, rather than guessing.
- Tell To contact Timmins WhatsApp +60 1136514727 Email to info@timmins-consulting.com
- Be concise, friendly, and practical — like a knowledgeable admissions counsellor, \
not a marketing brochure.
- When relevant, mention course names, formats (public/in-house/customized), and who \
the course is for, exactly as described in the context.
Context:
{{context}}
"""


# ----------------------------------------------------------------------------
# Secrets / config helpers
# ----------------------------------------------------------------------------
def get_groq_api_key() -> str:
    """Reads the key from Streamlit Cloud Secrets first, then env var.
    Never surfaced in the UI — end users can't see or change this."""
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# ----------------------------------------------------------------------------
# Knowledge base loading (auto, no user interaction)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_folder_documents(folder_path: str) -> list[Document]:
    docs: list[Document] = []
    if not os.path.isdir(folder_path):
        return docs
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            if fname.lower().endswith(".pdf"):
                from langchain_community.document_loaders import PyPDFLoader
                docs.extend(PyPDFLoader(fpath).load())
            elif fname.lower().endswith(".txt"):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
        except Exception as e:
            print(f"Could not read {fname}: {e}")
    return docs


@st.cache_resource(show_spinner=False)
def get_knowledge_base():
    """Cached once per running app instance (all visitors share it).

    Prefers a pre-built FAISS_INDEX_DIR (fast, reliable). Falls back to
    scraping KNOWLEDGE_URLS + KNOWLEDGE_FOLDER live if no index was committed.
    """
    embeddings = get_embeddings()

    if os.path.isdir(FAISS_INDEX_DIR):
        try:
            return FAISS.load_local(
                FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Could not load pre-built index ({e}); building live instead.")

    docs: list[Document] = []
    if KNOWLEDGE_URLS:
        loader = WebBaseLoader(KNOWLEDGE_URLS)
        loader.requests_kwargs = {"timeout": 15}
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"URL load error: {e}")
    docs.extend(load_folder_documents(KNOWLEDGE_FOLDER))

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


# ----------------------------------------------------------------------------
# RAG chain (langchain-core only — no langchain.chains dependency)
# ----------------------------------------------------------------------------
class SimpleRagChain:
    def __init__(self, llm, retriever, vendor_name):
        self.retriever = retriever

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given the chat history and the latest user question, rewrite it as "
                    "a standalone question that can be understood without the chat "
                    "history. Only output the rewritten question, nothing else. If no "
                    "rewrite is needed, return the question unchanged.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.contextualize_chain = contextualize_prompt | llm | StrOutputParser()

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_TEMPLATE.format(vendor_name=vendor_name)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.qa_chain = qa_prompt | llm | StrOutputParser()

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        if chat_history:
            standalone_question = self.contextualize_chain.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
        else:
            standalone_question = user_input

        docs = self.retriever.invoke(standalone_question)
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', d.metadata.get('title', 'unknown'))}]\n{d.page_content}"
            for d in docs
        )

        answer = self.qa_chain.invoke(
            {"input": user_input, "chat_history": chat_history, "context": context}
        )
        return {"answer": answer, "context": docs}


@st.cache_resource(show_spinner=False)
def build_rag_chain(_vectorstore, groq_api_key, model_name, vendor_name):
    # Leading underscore on _vectorstore tells st.cache_resource not to hash it
    # (FAISS objects aren't hashable) — it's still cached correctly by the
    # other (hashable) arguments.
    llm = ChatGroq(api_key=groq_api_key, model=model_name, temperature=0.2)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    return SimpleRagChain(llm, retriever, vendor_name)


# ----------------------------------------------------------------------------
# Streamlit UI — end users can ONLY chat. No setup, no keys, no uploads.
# ----------------------------------------------------------------------------
st.set_page_config(page_title=f"{VENDOR_NAME} Course Bot", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    .app-header {
        padding: 1.1rem 1.4rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0f3d3e 0%, #14532d 100%);
        color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { color: #ffffff; margin: 0; font-size: 1.6rem; }
    .app-header p { color: #d9f2ea; margin: 0.3rem 0 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

USER_AVATAR = "🧑‍🎓"
BOT_AVATAR = "🎓"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header(f"🎓 {VENDOR_NAME}")
    st.caption("Ask me anything about our courses, formats, prerequisites, or HRDC claimability.")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.markdown(
    f"""
    <div class="app-header">
        <h1>🎓 Course Enquiry Chatbot</h1>
        <p>Ask about courses, formats, prerequisites, or HRDC claimability for <b>{VENDOR_NAME}</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

groq_api_key = get_groq_api_key()
if not groq_api_key:
    st.error(
        "This assistant isn't fully configured yet. (Admin: add GROQ_API_KEY under "
        "App settings → Secrets in Streamlit Cloud.)"
    )
    st.stop()

with st.spinner("Warming up the course assistant..."):
    vectorstore = get_knowledge_base()

if vectorstore is None:
    st.error(
        "Could not load the course knowledge base. (Admin: check KNOWLEDGE_URLS / "
        "knowledge/ folder, or run build_index.py and commit faiss_index/.)"
    )
    st.stop()

rag_chain = build_rag_chain(vectorstore, groq_api_key, GROQ_MODEL, VENDOR_NAME)

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(msg.content)

user_input = st.chat_input("e.g. Is the Embedded Linux course beginner-friendly?")

if user_input:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            result = rag_chain.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history}
            )
            answer = result["answer"]
            st.markdown(answer)

            with st.expander("📚 Sources used"):
                seen = set()
                for doc in result.get("context", []):
                    src = doc.metadata.get("source", doc.metadata.get("title", "unknown"))
                    if src not in seen:
                        seen.add(src)
                        st.write(f"- {src}")

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
