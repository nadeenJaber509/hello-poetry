"""Inspect similarity scores for questions without documented answers."""

import csv
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from hello_poetry.store import query_collection

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
