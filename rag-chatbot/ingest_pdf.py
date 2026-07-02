"""
Ingests all PDFs in data/pdfs/ into the 'guides' Chroma collection.
Applies RecursiveCharacterTextSplitter to break each document into chunks.
Run once (or after adding/changing PDFs): python ingest_pdf.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_DIR    = os.path.join("data", "pdfs")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200


def main():
    pdf_paths = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}/")
        return

    print(f"Found {len(pdf_paths)} PDF(s): {[os.path.basename(p) for p in pdf_paths]}")

    all_pages = []
    for path in pdf_paths:
        print(f"Loading {os.path.basename(path)}...")
        loader = PyPDFLoader(path)
        pages = loader.load()
        print(f"  {len(pages)} pages loaded.")
        all_pages.extend(pages)

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(all_pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", "guide"))
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()