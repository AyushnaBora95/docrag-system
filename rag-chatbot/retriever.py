"""
Retriever for the 'guides' Chroma collection (PDFs).
Combines vector search (semantic) with BM25 (keyword) for hybrid retrieval,
then re-ranks the merged results with a cross-encoder for precision.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def build_retriever(k_guides: int = 15, k_final: int = 5) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vector_retriever = guides_store.as_retriever(search_kwargs={"k": k_guides})

    raw = guides_store.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(raw["documents"], raw["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k_guides

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    def retrieve(query: str) -> list[Document]:
        candidates = hybrid_retriever.invoke(query)
        if not candidates:
            return []

        cross_encoder = get_cross_encoder()
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = cross_encoder.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:k_final]]

    return RunnableLambda(retrieve)


if __name__ == "__main__":
    retriever = build_retriever()
    query = "What is the battery voltage and endurance of the 10 inch FPV drone?"
    docs = retriever.invoke(query)
    print(f"\n--- Retrieved {len(docs)} chunks for query: {query} ---")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1} (source: {doc.metadata.get('source')}):\n{doc.page_content[:300]}")