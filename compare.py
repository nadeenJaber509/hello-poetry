
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "task6" / "data" / "test_comments.csv"
MODEL_NAME = "all-MiniLM-L6-v2"


def keyword_search(comments, query, top_k=3):
    """Find comments containing query words, ignoring capitalization."""
    query_words = query.lower().split()

    if not query_words or top_k <= 0:
        return []

    results = []

    for index, comment in enumerate(comments):
        comment_lower = str(comment).lower()
        matches = sum(
            1 for word in query_words if word in comment_lower
        )
        score = matches / len(query_words)

        if score > 0:
            results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def semantic_search(comments, query, model, embeddings, top_k=3):
    """Find comments with similar meanings using embeddings."""
    if not query.strip() or top_k <= 0:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(index), float(scores[index]))
        for index in top_indices
    ]


def main():
    df = pd.read_csv(DATA_PATH)
    comments = df["comment"].fillna("").tolist()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        comments,
        normalize_embeddings=True,
    )

    queries = [
    "slow delivery",
    "broken product",
    "helpful customer service",
    "money back problem",
    "good value",
    "quantum physics",
    "invoice",
    "late arrival",
]
    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")

        print("\nKeyword Search:")
        keyword_results = keyword_search(comments, query)

        if not keyword_results:
            print("No matching comments found.")
        else:
            for index, score in keyword_results:
                print(
                    f"{df.iloc[index]['id']} | "
                    f"{comments[index]} | "
                    f"{df.iloc[index]['label']} | "
                    f"{score:.4f}"
                )

        print("\nSemantic Search:")
        semantic_results = semantic_search(
            comments,
            query,
            model,
            embeddings,
        )

        for index, score in semantic_results:
            print(
                f"{df.iloc[index]['id']} | "
                f"{comments[index]} | "
                f"{df.iloc[index]['label']} | "
                f"{score:.4f}"
            )


if __name__ == "__main__":
    main()
