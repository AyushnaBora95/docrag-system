"""
Ingests all PDFs in data/pdfs/ into the 'guides' Chroma collection.
Uses UnstructuredPDFLoader (fast strategy) for better structure detection,
then SemanticChunker to break each document into semantically coherent chunks.
Run once (or after adding/changing PDFs): python ingest_pdf.py
Also exposes ingest_single_pdf() for use by the admin upload feature in app.py.
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import glob
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_DIR    = os.path.join("data", "pdfs")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def load_pdf(path):
    loader = UnstructuredPDFLoader(path, mode="single", strategy="fast")
    return loader.load()


def chunk_pages(pages, embeddings):
    splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=75,

    )
    return splitter.split_documents(pages)


def ingest_single_pdf(path: str) -> int:
    """Ingest one PDF into the existing Chroma store. Returns number of chunks added."""
    embeddings = get_embeddings()

    pages = load_pdf(path)
    chunks = chunk_pages(pages, embeddings)
    for i, chunk in enumerate(chunks):
      chunk.metadata = {
        "source": os.path.basename(path),
        "chunk_index": i,
    }

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


def main():
    pdf_paths = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}/")
        return

    print(f"Found {len(pdf_paths)} PDF(s): {[os.path.basename(p) for p in pdf_paths]}")

    all_pages = []
    for path in pdf_paths:
        print(f"Loading {os.path.basename(path)}...")
        pages = load_pdf(path)
        print(f"  {len(pages)} elements loaded.")
        all_pages.extend(pages)

    embeddings = get_embeddings()

    print("Chunking semantically...")
    chunks = []
    for path in pdf_paths:
        doc_pages = [p for p in all_pages if p.metadata.get("source") == path]
        doc_chunks = chunk_pages(doc_pages, embeddings)
        chunks.extend(doc_chunks)

    for i, chunk in enumerate(chunks):
      source = chunk.metadata.get("source", "guide")
      chunk.metadata = {
        "source": os.path.basename(source),
        "chunk_index": i,
    }

    print(f"  {len(chunks)} chunks produced.")

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