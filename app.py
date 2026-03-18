import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dynamic Document AI", page_icon="📄", layout="wide")
st.title("📄 Dynamic RAG Assistant")
st.markdown("Upload any PDF and instantly ask questions about it!")
st.divider()

@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation", 
        model="google/flan-t5-base", 
        max_length=256, 
        temperature=0.1 
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = load_llm()
embeddings = load_embeddings()

# --- Sidebar for File Upload ---
st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # 1. Process the uploaded file
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Build in-memory database for this specific file
        db = Chroma.from_documents(chunks, embeddings)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}), 
            return_source_documents=True
        )
        
        st.sidebar.success("✅ PDF processed successfully!")

    # 2. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the uploaded document..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                response = qa_chain({"query": prompt})
                answer = response["result"]
                
                st.markdown(answer)
                
                with st.expander("📄 View Source Document Snippets"):
                    for idx, doc in enumerate(response["source_documents"]):
                        st.write(f"**Source {idx+1}:** Page {doc.metadata.get('page', 'Unknown')}")
                        st.write(f"*{doc.page_content}*")
                        st.divider()
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please upload a PDF file in the sidebar to start.")