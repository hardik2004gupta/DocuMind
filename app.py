import streamlit as st
import time
import tempfile

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="DocuMind", layout="wide")
st.title("🧠 DocuMind - Chat with your Documents")

# -------------------- API KEY --------------------
groq_api_key = st.text_input("🔑 Enter your Groq API Key", type="password")

# -------------------- MODEL SELECTION --------------------
mode = st.radio("Model Input", ["Select", "Custom"])

if mode == "Select":
    model_name = st.selectbox(
        "Choose model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]
    )
else:
    model_name = st.text_input("Enter model name")

st.caption("💡 8B = fast ⚡ | 70B = smarter 🧠")

# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer ONLY from the context.
    If not found, say:
    "Answer not available in documents."

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# -------------------- EMBEDDINGS --------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------- PROCESS FILES --------------------
@st.cache_resource
def process_documents(file_bytes_list):
    documents = []

    for file_bytes in file_bytes_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,   # smaller = faster
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents[:15])  # limit docs

    vectors = FAISS.from_documents(chunks, load_embeddings())
    return vectors

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader(
    "📤 Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------------------- PROCESS BUTTON --------------------
if st.button("📚 Process Documents"):
    if not groq_api_key:
        st.warning("Enter API key")
    elif not uploaded_files:
        st.warning("Upload PDFs")
    else:
        file_bytes = [file.read() for file in uploaded_files]

        with st.spinner("Processing..."):
            st.session_state.vectors = process_documents(file_bytes)

        st.success("Ready ✅")

# -------------------- QUERY --------------------
query = st.text_input("🔍 Ask your question")

if query:
    if "vectors" not in st.session_state:
        st.warning("Upload + process docs first")
        st.stop()

    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name
        )

        chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()

        rag = create_retrieval_chain(retriever, chain)

        start = time.time()
        response = rag.invoke({'input': query})
        end = time.time()

        st.subheader("🧠 Answer")
        st.write(response['answer'])

        st.caption(f"⏱ {end-start:.2f}s")

        with st.expander("📄 Context"):
            for doc in response['context']:
                st.write(doc.page_content)
                st.write("---")

    except Exception as e:
        st.error(str(e))