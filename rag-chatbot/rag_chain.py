"""
Builds the RAG chain with conversation memory:
  merged retriever → prompt (with history) → Qwen3-32B on Groq → string output
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from ingest_pdf import MAIN_COLLECTION
from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about telecom \
and drone documentation using ONLY the context provided below.

Do NOT use any outside knowledge, even if you know the answer from elsewhere. \
Base your answer strictly on the context.

Do NOT reference specifications from a different product/model than the one asked \
about, even as a comparison or "for reference." If the specific product's data isn't \
in the context, say so plainly — do not substitute or hint at another product's specs.

If the context does not contain enough information to answer the question, respond \
exactly with: "I don't have that information in the provided documents." Do not \
attempt to answer from general knowledge or other products in that case.

Context:
{context}
"""

# Prompt used to rewrite a follow-up question into a standalone question
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given a conversation history and a follow-up question, rewrite the \
follow-up question to be a standalone question that includes all necessary context \
from the history. Do not answer the question — only rewrite it.

If the follow-up question is already standalone (doesn't depend on prior context), \
return it unchanged.

Respond with ONLY the rewritten question, nothing else."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "NO_RELEVANT_CONTEXT_FOUND"
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),  # history goes here
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | _format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain