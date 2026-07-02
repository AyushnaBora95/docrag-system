import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import build_chain

load_dotenv()

SAMPLE_QUESTIONS = [
    "What is covered in chapter 2?",
    "Explain the first exercise in maths",
    "What are the key topics in software engineering chapter 1?",
    "Give me a summary of the digital logic questions",
]

st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="centered",
)

@st.cache_resource
def get_chain():
    return build_chain()

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Study Assistant")
    st.caption("Powered by RAG · Qwen3-32B on Groq")
    st.divider()

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Study Assistant")
st.caption("Ask me anything about your notes and course material.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve question from chat input or sidebar button
question = st.chat_input("Ask a question about your notes…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        chain = get_chain()
        response = st.write_stream(
            chain.stream({
                "question": question,
                "chat_history": st.session_state.chat_history,
            })
        )

    # Save to display history and LangChain message history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))