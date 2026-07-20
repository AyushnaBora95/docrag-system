from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

from ingest_pdf import CHROMA_DIR, MAIN_COLLECTION, get_embeddings


def build_retriever(
    collection_name: str = MAIN_COLLECTION,
    k: int = 5,
):
    """Return a retriever restricted to one Chroma collection."""

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    stored_data = vectorstore.get(include=["documents"])

    # Do not send unrelated answers when this product has no PDFs.
    if not stored_data.get("documents"):
        return RunnableLambda(lambda _: [])

    return vectorstore.as_retriever(search_kwargs={"k": k})