"""Store and retrieve document chunks using ChromaDB."""

from pathlib import Path

import chromadb

DEFAULT_DB_PATH = "chroma_db"
BATCH_SIZE = 100


def build_collection(
    name: str,
    chunks: list[dict],
    model,
    path: str | Path = DEFAULT_DB_PATH,
):
    """Build a persistent Chroma collection from document chunks."""
    client = chromadb.PersistentClient(path=str(path))

    # Rebuild from scratch to avoid duplicate or stale chunks.
    try:
        client.delete_collection(name=name)
    except chromadb.errors.NotFoundError:
        pass

    collection = client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
    )

    # Chroma cannot store empty documents as useful embeddings.
    valid_chunks = [chunk for chunk in chunks if chunk["text"].strip()]

    for start in range(0, len(valid_chunks), BATCH_SIZE):
        batch = valid_chunks[start : start + BATCH_SIZE]

        texts = [chunk["text"] for chunk in batch]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in batch
            ],
        )

    return collection


def query_collection(
    name: str,
    query: str,
    model,
    k: int = 5,
    path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Retrieve the top-k chunks ranked by cosine similarity."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not query.strip():
        return []

    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(name=name)

    if collection.count() == 0:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append(
            {
                "id": chunk_id,
                "text": text,
                "source": metadata["source"],
                "title": metadata["title"],
                "chunk_index": metadata["chunk_index"],
                "similarity": 1.0 - float(distance),
            }
        )

    return sorted(
        matches,
        key=lambda match: match["similarity"],
        reverse=True,
    )
