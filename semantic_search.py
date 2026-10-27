
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "task6" / "data" / "test_comments.csv"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_search_resources():
    """Load comments, model, and cached embeddings."""
    df = pd.read_csv(DATA_PATH)
    comments = df["comment"].fillna("").tolist()

    model = SentenceTransformer(MODEL_NAME)

    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)

        # Recalculate if the number of comments has changed.
        if embeddings.shape != (len(comments), 384):
            embeddings = model.encode(
                comments,
                normalize_embeddings=True,
            )
            np.save(EMBEDDINGS_PATH, embeddings)
    else:
        embeddings = model.encode(
            comments,
            normalize_embeddings=True,
        )
        np.save(EMBEDDINGS_PATH, embeddings)

    return df, model, embeddings


def semantic_search(
    query,
    df,
    embeddings,
    model,
    top_k=5,
    min_score=0.30,
):
    """Return the most similar comments above the minimum score."""
    if not query or not query.strip():
        return pd.DataFrame(
            columns=["id", "comment", "label", "similarity"]
        )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1]

    top_indices = [
        i for i in top_indices if scores[i] >= min_score
    ][:top_k]

    if not top_indices:
        return pd.DataFrame(
            columns=["id", "comment", "label", "similarity"]
        )

    results = df.iloc[top_indices][
        ["id", "comment", "label"]
    ].copy()

    results["similarity"] = scores[top_indices]
    return results


def main():
    df, model, embeddings = load_search_resources()

    queries = [
        "slow delivery",
        "broken product",
        "helpful customer service",
        "money back problem",
        "good value",
        "quantum physics",
    ]

    for query in queries:
        print(f"\nSearch query: {query}\n")

        results = semantic_search(
            query,
            df,
            embeddings,
            model,
            top_k=5,
            min_score=0.30,
        )

        if results.empty:
            print("No good match found.")
        else:
            print(results.to_string(index=False))


if __name__ == "__main__":
    main()
