import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

def build_multi_doc_database(source_folder=".", db_directory="./chroma_db"):
    # The "." tells it to scan the current folder where this script lives
    print(f"📂 Scanning current folder for PDFs...")
    
    loader = PyPDFDirectoryLoader(source_folder)
    documents = loader.load()
    
    if not documents:
        print("❌ No PDFs found. Please make sure your PDF files are in this folder.")
        return

    print(f"📄 Loaded {len(documents)} total pages from your PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} searchable data chunks.")

    print("🧠 Loading local embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Saving to local vector database...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )
    print("✅ Database built successfully from your main folder!")

if __name__ == "__main__":
    build_multi_doc_database()