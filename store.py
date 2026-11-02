
import math
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_documents
from documents import load_documents

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = BASE_DIR / "data" / "chroma_db"
DOCS_PATH = BASE_DIR / "data" / "poetry_docs"

COLLECTION_NAME = "poetry_1500_300"

CHUNK_SIZE = 1500
OVERLAP = 300


# --------------------------------------------------
# Build or load vector store
# --------------------------------------------------

def build_vector_store(
    folder=None,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
):
    if folder is None:
        folder = DOCS_PATH

    folder = Path(folder).resolve()
    DB_PATH.mkdir(parents=True, exist_ok=True)

    if not folder.exists():
        raise FileNotFoundError(
            f"Poetry documentation folder not found: {folder}"
        )

    collection_name = f"poetry_{chunk_size}_{overlap}"

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    model = SentenceTransformer(MODEL_NAME)

    # Load and prepare documents.
    documents = load_documents(str(folder))

    if not documents:
        raise ValueError(
            f"No Markdown documents found in: {folder}"
        )

    chunks = chunk_documents(
        documents,
        chunk_size,
        overlap,
    )

    if not chunks:
        raise ValueError(
            "No chunks were created from the documentation."
        )

    expected_count = len(chunks)

    # Get or create the collection.
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    current_count = collection.count()

    print("Collection:", collection_name)
    print("Existing chunks:", current_count)
    print("Expected chunks:", expected_count)

    # Rebuild if the collection is empty or incomplete.
    if current_count != expected_count:

        if current_count > 0:
            print("Incomplete collection found. Rebuilding...")

            client.delete_collection(
                name=collection_name
            )

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print("Building collection:", collection_name)

        batch_size = 100

        for start in range(0, expected_count, batch_size):

            batch = chunks[start:start + batch_size]

            texts = [
                chunk["text"]
                for chunk in batch
            ]

            embeddings = model.encode(
                texts
            ).tolist()

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

        print("Collection built successfully.")

    # Get a fresh collection reference after building.
    collection = client.get_collection(
        name=collection_name
    )

    final_count = collection.count()

    if final_count != expected_count:
        raise RuntimeError(
            f"Collection is incomplete. "
            f"Expected {expected_count} chunks, "
            f"but found {final_count}."
        )

    print("Final collection count:", final_count)
    print("Database path:", DB_PATH)

    return collection, model


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------

def search_documents(
    query,
    collection,
    model,
    top_k=5,
):
    collection_count = collection.count()

    if collection_count == 0:
        return []

    query_embedding = model.encode(
        [query]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    if not results["documents"]:
        return matches

    for i, text in enumerate(results["documents"][0]):

        cosine_distance = results["distances"][0][i]
        similarity = 1 - cosine_distance

        metadata = results["metadatas"][0][i]

        matches.append({
            "text": text,
            "source": metadata["source"],
            "title": metadata["title"],
            "similarity": similarity,
        })

    return matches


# --------------------------------------------------
# Keyword Search
# --------------------------------------------------

def keyword_search(
    query,
    collection,
    top_k=5,
):
    normalized_query = query.casefold()

    normalized_query = re.sub(
        r"[^\w\s]",
        " ",
        normalized_query,
    ).strip()

    query_words = set(
        normalized_query.split()
    )

    stop_words = {
        "what", "which", "where", "when", "how",
        "is", "are", "the", "a", "an", "and",
        "to", "of", "in", "on", "for", "with",
        "does", "do", "can", "be",
    }

    query_words -= stop_words

    if not query_words:
        return []

    stored = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    matches = []

    for i, text in enumerate(stored["documents"]):

        normalized_text = text.casefold()

        normalized_text = re.sub(
            r"[^\w\s]",
            " ",
            normalized_text,
        )

        text_words = set(
            normalized_text.split()
        )

        matched_words = query_words.intersection(
            text_words
        )

        if matched_words:

            phrase_bonus = (
                5 if normalized_query in normalized_text else 0
            )

            score = len(matched_words) + phrase_bonus

            metadata = stored["metadatas"][i]

            matches.append({
                "text": text,
                "source": metadata["source"],
                "title": metadata["title"],
                "keyword_score": score,
            })

    matches.sort(
        key=lambda item: item["keyword_score"],
        reverse=True,
    )

    return matches[:top_k]


# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------

def cosine_similarity(
    vector_a,
    vector_b,
):
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# --------------------------------------------------
# Hybrid Search
# --------------------------------------------------

def hybrid_search(
    query,
    collection,
    model,
    top_k=5,
    min_similarity=None,
):
    if min_similarity is not None and not 0 <= min_similarity <= 1:
        raise ValueError(
            "min_similarity must be between 0 and 1."
        )

    collection_count = collection.count()

    if collection_count == 0:
        return []

    candidate_count = min(
        max(top_k * 4, 20),
        collection_count,
    )

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
    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):
        key = (
            result["source"],
            result["text"],
        )

        combined_scores[key] = {
            **result,
            "rrf_score": 1 / (60 + rank),
        }

    # Find keyword results missing from semantic results.
    missing_similarity = []

    for result in keyword_results:

        key = (
            result["source"],
            result["text"],
        )

        if key not in combined_scores:
            missing_similarity.append(result)

    # Calculate semantic similarity for keyword-only results.
    if missing_similarity:

        query_embedding = model.encode(
            [query]
        ).tolist()[0]

        texts = [
            result["text"]
            for result in missing_similarity
        ]

        text_embeddings = model.encode(
            texts
        ).tolist()

        for result, embedding in zip(
            missing_similarity,
            text_embeddings,
        ):

            result["similarity"] = cosine_similarity(
                query_embedding,
                embedding,
            )

    # Add keyword results and combine RRF scores.
    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):

        key = (
            result["source"],
            result["text"],
        )

        if key not in combined_scores:

            combined_scores[key] = {
                **result,
                "rrf_score": 0,
            }

        combined_scores[key]["rrf_score"] += (
            1 / (60 + rank)
        )

    # Apply minimum similarity threshold.
    if min_similarity is not None:

        combined_scores = {
            key: result
            for key, result in combined_scores.items()
            if result.get("similarity", 0) >= min_similarity
        }

    # Sort by combined RRF score.
    ranked_results = sorted(
        combined_scores.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:top_k]


# --------------------------------------------------
# Local Testing
# --------------------------------------------------

if __name__ == "__main__":

    collection, model = build_vector_store()

    query = "Which command creates a new Poetry project?"

    print("\nKeyword Search")

    for index, result in enumerate(
        keyword_search(query, collection, top_k=5),
        start=1,
    ):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Keyword Score:", result["keyword_score"])
        print("Text:", result["text"][:300])

    print("\nSemantic Search")

    for index, result in enumerate(
        search_documents(query, collection, model, top_k=5),
        start=1,
    ):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Title:", result["title"])
        print("Similarity:", round(result["similarity"], 4))
        print("Text:", result["text"][:300])

    print("\nHybrid Search")

    for index, result in enumerate(
        hybrid_search(
            query,
            collection,
            model,
            top_k=5,
            min_similarity=None,
        ),
        start=1,
    ):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Title:", result["title"])
        print("Similarity:", round(result["similarity"], 4))
        print("RRF Score:", round(result["rrf_score"], 4))
        print("Text:", result["text"][:300])
