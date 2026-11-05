
"""Educational Streamlit page for Poetry document search."""

import sys
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# Make project-level modules available from the pages directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hello_poetry.chunking import chunk_documents
from hello_poetry.documents import load_documents
from hello_poetry.store import build_collection, query_collection

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
        raise FileNotFoundError(
            "Poetry documentation files were not found."
        )

    with st.spinner(f"Building collection: {name}..."):
        documents = load_documents(DOCS_DIR)
        chunks = chunk_documents(
            documents,
            chunk_size,
            overlap,
        )

        if not chunks:
            raise ValueError(
                "No document chunks were generated."
            )

        collection = build_collection(
            name=name,
            chunks=chunks,
            model=model,
            path=DB_DIR,
        )

        return collection.count()


# --------------------------------------------------
# Educational introduction
# --------------------------------------------------

st.caption("TASK 08 · LANGUAGE AI & RETRIEVAL")
st.title("Document Search")

st.write(
    "Search Poetry documentation using embeddings "
    "and a vector database."
)

with st.container(border=True):
    st.markdown("#### What is Document Search?")
    st.write(
        "Long documents are divided into smaller pieces "
        "called chunks. Each chunk is converted into an "
        "embedding and stored in a vector database. "
        "When you ask a question, the system finds the "
        "chunks that are most similar to your query."
    )

col1, col2 = st.columns(2, gap="medium")

with col1, st.container(border=True):
    st.markdown("**Try it**")
    st.write(
        'Ask "How do I add a dependency using Poetry?" '
        "and compare the results using different "
        "chunking configurations."
    )

with col2, st.container(border=True):
    st.markdown("**What to notice**")
    st.write(
        "Observe how chunk size, overlap, Top-k, and "
        "similarity thresholds change the retrieved results."
    )

st.divider()


# --------------------------------------------------
# Interactive document search
# --------------------------------------------------

st.subheader("Interactive Demo")
st.caption("Searching Poetry 2.1.4 documentation")

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

search_clicked = st.button(
    "Search",
    type="primary",
)

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

            st.caption(
                f"Collection: {collection_name} | "
                f"Stored chunks: {count}"
            )

            results = query_collection(
                name=collection_name,
                query=query.strip(),
                model=model,
                k=top_k,
                path=DB_DIR,
            )

            filtered_results = [
                result
                for result in results
                if result["similarity"] >= min_similarity
            ]

            if not filtered_results:
                st.info("No good match found.")
            else:
                st.success(
                    f"Found {len(filtered_results)} matching chunks."
                )

                for rank, result in enumerate(
                    filtered_results,
                    start=1,
                ):
                    st.markdown(
                        f"**{rank}. {result['title']}**"
                    )

                    st.caption(
                        f"Source: {result['source']} | "
                        f"Chunk: {result['chunk_index']} | "
                        f"Similarity: {result['similarity']:.4f}"
                    )

                    with st.expander("View retrieved text"):
                        st.code(
                            result["text"],
                            language=None,
                        )

        except Exception as error:  # noqa: BLE001
            st.error(f"Document search failed: {error}")
            st.exception(error)
