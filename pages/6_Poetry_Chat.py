
"""Streamlit chat interface for the Poetry RAG assistant."""

from pathlib import Path

import chromadb
import streamlit as st
from groq import GroqError

from hello_poetry.chunking import chunk_documents
from hello_poetry.documents import load_documents
from hello_poetry.rag import answer_question, get_embedding_model
from hello_poetry.store import build_collection

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "poetry_docs"
DB_DIR = ROOT / "chroma_db"

COLLECTION_NAME = "poetry_800_150"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

st.set_page_config(
    page_title="Poetry RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# Collection setup
# --------------------------------------------------

def ensure_rag_collection():
    """Use the existing ChromaDB collection or build it if missing."""
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)

        if collection.count() > 0:
            return

    except chromadb.errors.NotFoundError:
        pass

    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(
            "Poetry documentation files were not found."
        )

    with st.spinner("Preparing Poetry documentation..."):
        documents = load_documents(DOCS_DIR)

        chunks = chunk_documents(
            documents,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError(
                "No document chunks were generated."
            )

        model = get_embedding_model()

        collection = build_collection(
            name=COLLECTION_NAME,
            chunks=chunks,
            model=model,
            path=DB_DIR,
        )

        if collection.count() == 0:
            raise ValueError(
                "The ChromaDB collection is empty."
            )


# --------------------------------------------------
# Helper: display retrieved sources
# --------------------------------------------------

def display_sources(sources):
    """Display numbered sources and their similarity scores."""
    with st.expander("View retrieved sources"):
        if not sources:
            st.info("No documentation sources were retrieved.")
            return

        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{index}] {source['title']}**"
            )

            st.caption(
                f"File: {source['source']} | "
                f"Similarity: {source['similarity']:.4f}"
            )

            st.text(source["text"])

            if index < len(sources):
                st.divider()


# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("🤖 Poetry RAG Assistant")

st.caption(
    "Ask questions about Poetry using its documentation."
)


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

with st.sidebar:
    st.header("RAG Settings")

    k = st.slider(
        "Number of retrieved chunks (k)",
        min_value=1,
        max_value=8,
        value=3,
        help="Number of documentation chunks sent to the LLM.",
    )

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):
        st.session_state["rag_messages"] = []
        st.rerun()


# --------------------------------------------------
# Initialize chat history
# --------------------------------------------------

if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state["rag_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(
                message.get("sources", [])
            )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about Poetry..."
)

if question:
    # Save and display the user message.
    st.session_state["rag_messages"].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display the assistant response.
    with (
        st.chat_message("assistant"),
        st.spinner("Searching Poetry documentation..."),
    ):
        try:
            # Make sure the collection exists before retrieval.
            ensure_rag_collection()

            # Run the RAG pipeline.
            result = answer_question(
                question=question,
                collection_name=COLLECTION_NAME,
                k=k,
            )

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            display_sources(sources)

            # Save the assistant response and its sources.
            st.session_state["rag_messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except (
            GroqError,
            ValueError,
            RuntimeError,
            OSError,
            chromadb.errors.ChromaError,
        ) as exc:
            st.error(
                f"Could not generate an answer: {exc}"
            )

            st.session_state["rag_messages"].append({
                "role": "assistant",
                "content": (
                    "Sorry, an error occurred while processing "
                    "your question."
                ),
                "sources": [],
            })
