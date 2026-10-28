
from sentence_transformers import SentenceTransformer

from search import (
    MODEL_NAME,
    get_embeddings,
    keyword_search,
    load_comments,
    search,
)

QUERIES = [
    "slow delivery",
    "broken product",
    "helpful customer service",
    "money back problem",
    "good value",
    "quantum physics",
    "invoice",
    "late arrival",
]


def print_results(title, results, df):
    print(f"\n{title}")

    for result in results:
        row = df.iloc[result["index"]]

        print(
            f"ID {row['id']} | Score: {result['score']:.4f}"
        )
        print(f"  {result['comment']}")


def main():
    df = load_comments()
    comments = df["comment"].fillna("").tolist()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_embeddings(comments, model)

    for query in QUERIES:
        print(f"\n{'=' * 55}")
        print(f"QUERY: {query}")

        semantic_results = search(
            query, comments, embeddings, model, k=3
        )

        keyword_results = keyword_search(
            query, comments, k=3
        )

        print_results("SEMANTIC SEARCH", semantic_results, df)
        print_results("KEYWORD SEARCH", keyword_results, df)


if __name__ == "__main__":
    main()
