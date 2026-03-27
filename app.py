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
st.set_page_config(page_title="DocuMind AI", page_icon="", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4CAF50; color: white; }
    .stTextInput>div>div>input { border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# -------------------- SIDEBAR CONFIG --------------------
with st.sidebar:
    st.title("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get your key at console.groq.com")
    
    if groq_api_key:
        st.success("API Key Detected", icon="")
    else:
        st.warning("Please enter your API Key")

    st.divider()
    
    st.subheader("Model Settings")
    mode = st.radio("Input Mode", ["Select Preset", "Custom Name"])
    
    if mode == "Select Preset":
        model_name = st.selectbox(
            "Choose Model",
            ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
        )
    else:
        model_name = st.text_input("Enter Model ID", value="llama-3.1-8b-instant")
    
    st.info(f"Active Model: **{model_name}**")
    
    st.divider()
    st.caption("Made with ❤️ by DocuMind")

# -------------------- MAIN UI --------------------
st.title("DocuMind")
st.markdown("#### *Transform your PDFs into an interactive knowledge base*")

# Layout columns for processing
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF)",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDFs to start chatting."
    )

# -------------------- CORE LOGIC (UNCHANGED) --------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_documents(file_bytes_list):
    documents = []
    for file_bytes in file_bytes_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents[:15])
    vectors = FAISS.from_documents(chunks, load_embeddings())
    return vectors, len(chunks)

# -------------------- PROCESSING BUTTON --------------------
with col2:
    st.write("##") # Alignment spacer
    if st.button("Index Documents"):
        if not groq_api_key:
            st.error("Missing API Key")
        elif not uploaded_files:
            st.error("No files uploaded")
        else:
            with st.spinner("Analyzing documents..."):
                file_bytes = [f.read() for f in uploaded_files]
                vectors, count = process_documents(file_bytes)
                st.session_state.vectors = vectors
                st.session_state.chunk_count = count
                st.success(f"Indexed {len(uploaded_files)} files into {count} chunks!")

# -------------------- CHAT INTERFACE --------------------
st.divider()

# Display stats if data exists
if "vectors" in st.session_state:
    m1, m2 = st.columns(2)
    m1.metric("Status", "Ready", "Knowledge Base Active")
    m2.metric("Chunks Processed", st.session_state.chunk_count)

query = st.chat_input("Ask a question about your documents...")

if query:
    if "vectors" not in st.session_state:
        st.error("Please process your documents first!")
    else:
        # User Message
        with st.chat_message("user"):
            st.write(query)

        # AI Message
        with st.chat_message("assistant", avatar=""):
            try:
                llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
                
                prompt_template = ChatPromptTemplate.from_template(
                    "Answer ONLY from context. If not found, say 'Not in docs.'\n<context>{context}</context>\nQuestion: {input}"
                )
                
                chain = create_stuff_documents_chain(llm, prompt_template)
                retriever = st.session_state.vectors.as_retriever()
                rag = create_retrieval_chain(retriever, chain)

                start = time.time()
                response = rag.invoke({'input': query})
                end = time.time()

                st.write(response['answer'])
                st.caption(f"Inference time: {end-start:.2f}s")

                with st.expander("View Source Context"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Source {i+1}:**")
                        st.info(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")