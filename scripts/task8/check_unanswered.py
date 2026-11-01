"""Inspect similarity scores for questions without documented answers."""

import csv

from sentence_transformers import SentenceTransformer

from store import query_collection


def main() -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("data/questions.csv", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))

    unanswered = [
        question for question in questions if not question["expected_source"].strip()
    ]

    for question in unanswered:
        print(f"\nQuestion: {question['question']}")

        results = query_collection(
            name="poetry_300_60",
            query=question["question"],
            model=model,
            k=3,
        )

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['source']} | Similarity: {result['similarity']:.4f}"
            )


if __name__ == "__main__":
    main()
