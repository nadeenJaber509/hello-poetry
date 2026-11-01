"""Streamlit interface for searching Poetry documentation."""

import sys
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# Make project-level modules available from the pages directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunking import chunk_documents
from documents import load_documents
from store import build_collection, query_collection

DOCS_DIR = ROOT / "data" / "poetry_docs"
DB_DIR = ROOT / "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

CONFIGURATIONS = {
    "300 characters / 0 overlap": (300, 0),
    "300 characters / 60 overlap": (300, 60),
    "800 characters / 0 overlap": (800, 0),
    "800 characters / 150 overlap": (800, 150),
    "1500 characters / 0 overlap": (1500, 0),
    "1500 characters / 300 overlap": (1500, 300),
}


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


def ensure_collection(name, chunk_size, overlap, model):
    """Open an existing collection or build it when missing."""
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(name=name)
        if collection.count() > 0:
            return collection.count()
    except chromadb.errors.NotFoundError:
        pass

    if not DOCS_DIR.is_dir():
        raise FileNotFoundError("Poetry documentation files were not found.")

    with st.spinner(f"Building collection: {name}..."):
        documents = load_documents(DOCS_DIR)
        chunks = chunk_documents(documents, chunk_size, overlap)

        if not chunks:
            raise ValueError("No document chunks were generated.")

        collection = build_collection(
            name=name,
            chunks=chunks,
            model=model,
            path=DB_DIR,
        )

        return collection.count()


st.title("Docs Search")
st.caption("Semantic search over Poetry 2.1.4 documentation")

st.write(
    "Search the documentation using sentence embeddings "
    "and a persistent ChromaDB vector database."
)

configuration = st.selectbox(
    "Chunking configuration",
    options=list(CONFIGURATIONS),
    index=3,
)

chunk_size, overlap = CONFIGURATIONS[configuration]
collection_name = f"poetry_{chunk_size}_{overlap}"

top_k = st.slider(
    "Number of results (Top-k)",
    min_value=1,
    max_value=10,
    value=5,
)

min_similarity = st.slider(
    "Minimum similarity",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
)

query = st.text_input(
    "Ask a question",
    placeholder="How can I add a dependency using Poetry?",
)

search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            model = load_model()

            count = ensure_collection(
                collection_name,
                chunk_size,
                overlap,
                model,
            )

            st.caption(f"Collection: {collection_name} | Stored chunks: {count}")

            results = query_collection(
                name=collection_name,
                query=query.strip(),
                model=model,
                k=top_k,
                path=DB_DIR,
            )

            filtered_results = [
                result for result in results if result["similarity"] >= min_similarity
            ]

            if not filtered_results:
                st.info("No good match found.")
            else:
                st.success(f"Found {len(filtered_results)} matching chunks.")

                for rank, result in enumerate(filtered_results, start=1):
                    st.markdown(f"**{rank}. {result['title']}**")

                    st.caption(
                        f"Source: {result['source']} | "
                        f"Chunk: {result['chunk_index']} | "
                        f"Similarity: {result['similarity']:.4f}"
                    )

                    with st.expander("View retrieved text"):
                        st.code(result["text"], language=None)

        except Exception as error:  # noqa: BLE001
            st.error(f"Document search failed: {error}")
            st.exception(error)
