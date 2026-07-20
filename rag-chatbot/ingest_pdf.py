import os
import hashlib
import re
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_store"
PDF_DIR = BASE_DIR / "data" / "pdfs"

# Normal / main chat uses this collection
MAIN_COLLECTION = "guides"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    return _embeddings


def product_collection_name(folder_name: str) -> str:
    """Create a safe Chroma collection name from a product-folder name."""
    slug = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")

    if not slug:
        slug = "unknown"

    return f"product_{slug}"[:60]


def load_pdf(pdf_path: str | Path):
    loader = UnstructuredPDFLoader(
        str(pdf_path),
        mode="single",
        strategy="fast",
    )
    return loader.load()


def chunk_pages(pages, embeddings):
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
    )
    return splitter.split_documents(pages)


def ingest_single_pdf(
    pdf_path: str | Path,
    collection_name: str = MAIN_COLLECTION,
    source_name: str | None = None,
) -> int:
    """Chunk and add one PDF to the specified collection."""
    pdf_path = Path(pdf_path)
    source_path = str(pdf_path.resolve())
    source = source_name or pdf_path.name

    with open(pdf_path, "rb") as file:
        document_id = hashlib.sha256(file.read()).hexdigest()

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    existing = vectorstore.get(
        where={"source_path": source_path},
        include=["metadatas"],
    )

    existing_ids = existing.get("ids", [])
    existing_metadata = existing.get("metadatas") or []

    if existing_ids:
        existing_hashes = {
            metadata.get("document_id")
            for metadata in existing_metadata
            if metadata
        }

        # PDF was already indexed and has not changed.
        if existing_hashes == {document_id}:
            return len(existing_ids)

        # File changed: remove its old chunks first.
        vectorstore.delete(ids=existing_ids)

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, embeddings)

    if not chunks:
        return 0

    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            "source": source,
            "source_path": source_path,
            "document_id": document_id,
            "chunk_index": index,
            "collection": collection_name,
        }

    vectorstore.add_documents(
        documents=chunks,
        ids=[f"{document_id}:{index}" for index in range(len(chunks))],
    )

    return len(chunks)


def ingest_product_folder(product_folder: str | Path) -> int:
    """
    Index all PDFs in one product folder into that product's private collection.
    """
    product_folder = Path(product_folder)
    collection_name = product_collection_name(product_folder.name)

    total_chunks = 0

    for pdf_path in sorted(product_folder.glob("*.pdf")):
        total_chunks += ingest_single_pdf(
            pdf_path=pdf_path,
            collection_name=collection_name,
            source_name=pdf_path.name,
        )

    return total_chunks


def main():
    """Index PDFs for the normal main chat from data/pdfs/."""
    if not PDF_DIR.exists():
        print(f"Folder does not exist: {PDF_DIR}")
        return

    total_chunks = 0

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        total_chunks += ingest_single_pdf(pdf_path)

    print(f"Finished indexing {total_chunks} chunks.")


if __name__ == "__main__":
    main()