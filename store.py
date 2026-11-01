
from pathlib import Path
import re
import math

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_documents
from documents import load_documents


MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "data/chroma_db"

COLLECTION_NAME = "poetry_1500_300"

CHUNK_SIZE = 1500
OVERLAP = 300


def build_vector_store(
    folder="data/poetry_docs",
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
):
    collection_name = f"poetry_{chunk_size}_{overlap}"

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(MODEL_NAME)

    if collection.count() > 0:
        print("Existing collection found:", collection_name)
        print("Chunks stored:", collection.count())

        return collection, model

    print("Building collection:", collection_name)

    documents = load_documents(folder)
    chunks = chunk_documents(documents, chunk_size, overlap)

    print("Documents loaded:", len(documents))
    print("Chunks created:", len(chunks))

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts).tolist()

        ids = [
            f"{chunk['source']}-chunk-{start + i}"
            for i, chunk in enumerate(batch)
        ]

        metadatas = [
            {
                "source": chunk["source"],
                "title": chunk["title"],
            }
            for chunk in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print("Chunks stored:", collection.count())
    print("Database path:", Path(DB_PATH).resolve())

    return collection, model


def search_documents(query, collection, model, top_k=5):
    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        cosine_distance = results["distances"][0][i]
        similarity = 1 - cosine_distance

        matches.append({
            "text": text,
            "source": results["metadatas"][0][i]["source"],
            "title": results["metadatas"][0][i]["title"],
            "similarity": similarity,
        })

    return matches


def keyword_search(query, collection, top_k=5):
    normalized_query = query.casefold()
    normalized_query = re.sub(r"[^\w\s]", " ", normalized_query).strip()

    query_words = set(normalized_query.split())

    stop_words = {
        "what", "which", "where", "when", "how",
        "is", "are", "the", "a", "an", "and",
        "to", "of", "in", "on", "for", "with",
        "does", "do", "can", "be",
    }

    query_words -= stop_words

    stored = collection.get(
        include=["documents", "metadatas"]
    )

    matches = []

    for i, text in enumerate(stored["documents"]):
        normalized_text = text.casefold()
        normalized_text = re.sub(r"[^\w\s]", " ", normalized_text)

        text_words = set(normalized_text.split())
        matched_words = query_words.intersection(text_words)

        if matched_words:
            phrase_bonus = (
                5 if normalized_query in normalized_text else 0
            )

            score = len(matched_words) + phrase_bonus

            matches.append({
                "text": text,
                "source": stored["metadatas"][i]["source"],
                "title": stored["metadatas"][i]["title"],
                "keyword_score": score,
            })

    matches.sort(
        key=lambda item: item["keyword_score"],
        reverse=True,
    )

    return matches[:top_k]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def hybrid_search(
    query,
    collection,
    model,
    top_k=5,
    min_similarity=None,
):
    if min_similarity is not None:
        if not 0 <= min_similarity <= 1:
            raise ValueError(
                "min_similarity must be between 0 and 1."
            )

    candidate_count = max(top_k * 4, 20)

    semantic_results = search_documents(
        query=query,
        collection=collection,
        model=model,
        top_k=candidate_count,
    )

    keyword_results = keyword_search(
        query=query,
        collection=collection,
        top_k=candidate_count,
    )

    combined_scores = {}

    # Add semantic search results.
    for rank, result in enumerate(semantic_results, start=1):
        key = (result["source"], result["text"])

        combined_scores[key] = {
            **result,
            "rrf_score": 1 / (60 + rank),
        }

    # Find keyword results that are not already in semantic results.
    missing_similarity = []

    for result in keyword_results:
        key = (result["source"], result["text"])

        if key not in combined_scores:
            missing_similarity.append(result)

    # Calculate semantic similarity for keyword-only results.
    if missing_similarity:
        query_embedding = model.encode([query]).tolist()[0]

        texts = [
            result["text"] for result in missing_similarity
        ]

        text_embeddings = model.encode(texts).tolist()

        for result, embedding in zip(
            missing_similarity,
            text_embeddings,
        ):
            result["similarity"] = cosine_similarity(
                query_embedding,
                embedding,
            )

    # Add keyword search results and combine RRF scores.
    for rank, result in enumerate(keyword_results, start=1):
        key = (result["source"], result["text"])

        if key not in combined_scores:
            combined_scores[key] = {
                **result,
                "rrf_score": 0,
            }

        combined_scores[key]["rrf_score"] += 1 / (60 + rank)

    # Apply the minimum semantic similarity threshold.
    if min_similarity is not None:
        combined_scores = {
            key: result
            for key, result in combined_scores.items()
            if result["similarity"] >= min_similarity
        }

    # Sort by the combined RRF score.
    ranked_results = sorted(
        combined_scores.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:top_k]


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query = "Which command creates a new Poetry project?"

    # Test keyword search.
    keyword_results = keyword_search(
        query=query,
        collection=collection,
        top_k=20,
    )

    print("\nKeyword Search:", query)

    for index, result in enumerate(keyword_results, start=1):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Keyword Score:", result["keyword_score"])
        print("Text:", result["text"][:300])

    # Test semantic search.
    semantic_results = search_documents(
        query=query,
        collection=collection,
        model=model,
        top_k=5,
    )

    print("\nSemantic Search:", query)

    for index, result in enumerate(semantic_results, start=1):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Title:", result["title"])
        print("Similarity:", round(result["similarity"], 4))
        print("Text:", result["text"][:300])

    # Test hybrid search.
    hybrid_results = hybrid_search(
        query=query,
        collection=collection,
        model=model,
        top_k=5,
        min_similarity=None,
    )

    print("\nHybrid Search:", query)

    for index, result in enumerate(hybrid_results, start=1):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Title:", result["title"])
        print("Similarity:", round(result["similarity"], 4))
        print("RRF Score:", round(result["rrf_score"], 4))
        print("Text:", result["text"][:300])
