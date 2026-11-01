"""Run a real semantic search over Poetry documentation."""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from chunking import chunk_documents
from documents import load_documents
from store import build_collection, query_collection

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COLLECTION_NAME = "poetry_300_60"
MODEL_NAME = "all-MiniLM-L6-v2"
QUERY = "How do I add a dependency using Poetry?"


def main() -> None:
    # Load and chunk the documentation.
    documents = load_documents("data/poetry_docs")
    chunks = chunk_documents(documents, chunk_size=300, overlap=60)

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    # Load the same embedding model used in Task 7.
    model = SentenceTransformer(MODEL_NAME)

    # Build and save the vector collection.
    collection = build_collection(
        name=COLLECTION_NAME,
        chunks=chunks,
        model=model,
    )

    print(f"Stored chunks: {collection.count()}")

    # Search the saved collection.
    results = query_collection(
        name=COLLECTION_NAME,
        query=QUERY,
        model=model,
        k=3,
    )

    print(f"\nQuestion: {QUERY}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Title: {result['title']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Text:\n{result['text']}")


if __name__ == "__main__":
    main()
