
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from similarity import cosine_similarity

DATA_PATH = Path("data/test_comments.csv")
EMBEDDINGS_PATH = Path("embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_comments():
    """Read the comments from Task 6."""
    return pd.read_csv(DATA_PATH)


def get_embeddings(comments, model):
    """Load cached embeddings or generate and save them."""
    if EMBEDDINGS_PATH.exists():
        saved = np.load(EMBEDDINGS_PATH)

        if len(saved) == len(comments):
            return saved

    embeddings = model.encode(comments)
    np.save(EMBEDDINGS_PATH, embeddings)
    return embeddings


def search(query, comments, embeddings, model, k=5):
    """Return the top-k most semantically similar comments."""
    if not query.strip():
        return []

    query_vector = model.encode([query])[0]

    results = []

    for index, comment in enumerate(comments):
        score = cosine_similarity(
            query_vector,
            embeddings[index],
        )

        results.append({
            "index": index,
            "comment": comment,
            "score": score,
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:k]


def keyword_search(query, comments, k=5):
    """Rank comments by the number of matching query words."""
    if not query.strip():
        return []

    query_words = set(query.lower().split())
    results = []

    for index, comment in enumerate(comments):
        comment_words = set(comment.lower().split())

        score = len(query_words.intersection(comment_words))

        results.append({
            "index": index,
            "comment": comment,
            "score": score,
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:k]


def main():
    import sys

    if len(sys.argv) < 2:
        print('Usage: poetry run python search.py "your query"')
        return

    query = " ".join(sys.argv[1:])

    df = load_comments()

    text_column = (
        "text" if "text" in df.columns else "comment"
    )

    comments = df[text_column].fillna("").tolist()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_embeddings(comments, model)

    results = search(query, comments, embeddings, model)

    print(f"\nSearch query: {query}\n")

    for rank, result in enumerate(results, start=1):
        row = df.iloc[result["index"]]
        comment_id = row["id"] if "id" in df.columns else result["index"] + 1

        print(
            f"{rank}. ID {comment_id} "
            f"(score: {result['score']:.4f})"
        )
        print(f"   {result['comment']}\n")


if __name__ == "__main__":
    main()
