import os
import uuid
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from ingest_pdf import (
    MAIN_COLLECTION,
    ingest_product_folder,
    product_collection_name,
)
from rag_chain import build_chain

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_DIR = BASE_DIR / "data" / "products"

st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def get_chain(collection_name: str):
    return build_chain(collection_name)


def create_chat(
    title: str = "New chat",
    collection_name: str = MAIN_COLLECTION,
    product_name: str | None = None,
):
    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {
        "title": title,
        "messages": [],
        "chat_history": [],
        "collection_name": collection_name,
        "product_name": product_name,
    }

    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id: str):
    del st.session_state.chats[chat_id]

    if st.session_state.current_chat_id == chat_id:
        if st.session_state.chats:
            st.session_state.current_chat_id = next(
                iter(st.session_state.chats)
            )
        else:
            create_chat()


def make_title(question: str, max_length: int = 40) -> str:
    question = question.strip().replace("\n", " ")
    return (
        question
        if len(question) <= max_length
        else question[: max_length - 1].rstrip() + "…"
    )


# Session state
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if not st.session_state.chats:
    create_chat()

current_chat = st.session_state.chats[st.session_state.current_chat_id]


# Sidebar
with st.sidebar:
    st.title("📚 Study Assistant")
    st.caption("Powered by RAG · Qwen3-32B on Groq")

    if st.button("➕ New chat", use_container_width=True, type="primary"):
        create_chat()
        st.rerun()

    st.divider()
    st.markdown("**Chats**")

    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[chat_id]
        active = chat_id == st.session_state.current_chat_id

        left_column, right_column = st.columns([5, 1])

        with left_column:
            label = ("🟢 " if active else "") + chat["title"]

            if st.button(
                label,
                key=f"select_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()

        with right_column:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

    st.divider()

    # Same sidebar section as the old sample-question buttons.
    st.markdown("**Products**")
    st.caption("Select a product to start a chat using only its documents.")

    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    product_folders = sorted(
        folder
        for folder in PRODUCTS_DIR.iterdir()
        if folder.is_dir()
    )

    if not product_folders:
        st.info("Add product folders inside data/products/")

    for product_folder in product_folders:
        product_name = product_folder.name.replace("_", " ").replace("-", " ").title()

        if st.button(product_name, use_container_width=True):
            collection_name = product_collection_name(product_folder.name)

            with st.spinner(f"Loading {product_name} documents..."):
                ingest_product_folder(product_folder)

            create_chat(
                title=f"{product_name} chat",
                collection_name=collection_name,
                product_name=product_name,
            )
            st.rerun()


# Main page
st.title("Study Assistant")

if current_chat["product_name"]:
    st.caption(
        f"Product chat: **{current_chat['product_name']}** · "
        "Answers use only this product's PDF files."
    )
else:
    st.caption("Ask me anything about your notes and course material.")

for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your notes…")

if question:
    if not current_chat["messages"]:
        current_chat["title"] = make_title(question)

    current_chat["messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chain = get_chain(current_chat["collection_name"])

        response = st.write_stream(
            chain.stream(
                {
                    "question": question,
                    "chat_history": current_chat["chat_history"],
                }
            )
        )

    current_chat["messages"].append(
        {"role": "assistant", "content": response}
    )
    current_chat["chat_history"].append(HumanMessage(content=question))
    current_chat["chat_history"].append(AIMessage(content=response))
