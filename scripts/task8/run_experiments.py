"""Compare document chunking configurations using retrieval hit rates."""

import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer

from chunking import chunk_documents
from documents import load_documents
from store import build_collection, query_collection

DOCS_DIR = Path("data/poetry_docs")
QUESTIONS_FILE = Path("data/questions.csv")
RESULTS_DIR = Path("results/task8")

MODEL_NAME = "all-MiniLM-L6-v2"

CONFIGURATIONS = [
    (300, 0),
    (300, 60),
    (800, 0),
    (800, 150),
    (1500, 0),
    (1500, 300),
]


def load_questions(path: Path) -> list[dict]:
    """Read answerable evaluation questions from a CSV file."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"id", "question", "expected_source", "key_phrase"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Question CSV is missing required columns.")

        questions = list(reader)

    # No-answer questions are evaluated separately in Step 8.
    return [
        question
        for question in questions
        if question["expected_source"].strip() and question["key_phrase"].strip()
    ]


def is_hit(result: dict, question: dict) -> bool:
    """Check both source and expected phrase."""
    return (
        result["source"] == question["expected_source"].strip()
        and question["key_phrase"].strip().casefold() in result["text"].casefold()
    )


def calculate_hit_at_k(
    results: list[dict],
    question: dict,
    k: int,
) -> bool:
    """Return True if a correct chunk appears within the top-k."""
    return any(is_hit(result, question) for result in results[:k])


def save_failures(path: Path, failures: list[dict]) -> None:
    """Save questions whose correct chunk was not found in the top five."""
    fieldnames = [
        "id",
        "question",
        "expected_source",
        "key_phrase",
        "top_3_results",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents(DOCS_DIR)
    questions = load_questions(QUESTIONS_FILE)

    if not questions:
        raise ValueError("No answerable evaluation questions found.")

    print(f"Documents: {len(documents)}")
    print(f"Answerable questions: {len(questions)}")

    # Load the model once and reuse it across all six experiments.
    model = SentenceTransformer(MODEL_NAME)
    summaries = []

    for chunk_size, overlap in CONFIGURATIONS:
        name = f"poetry_{chunk_size}_{overlap}"

        print(f"\nRunning experiment: {name}")

        chunks = chunk_documents(documents, chunk_size, overlap)

        collection = build_collection(
            name=name,
            chunks=chunks,
            model=model,
        )

        hits = {1: 0, 3: 0, 5: 0}
        failures = []

        for question in questions:
            results = query_collection(
                name=name,
                query=question["question"],
                model=model,
                k=5,
            )

            for k in hits:
                if calculate_hit_at_k(results, question, k):
                    hits[k] += 1

            if not calculate_hit_at_k(results, question, 5):
                top_three = [
                    (
                        f"{result['source']}::{result['chunk_index']} "
                        f"(similarity={result['similarity']:.4f}) "
                        f"{result['text'][:150]!r}"
                    )
                    for result in results[:3]
                ]

                failures.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "expected_source": question["expected_source"],
                        "key_phrase": question["key_phrase"],
                        "top_3_results": " | ".join(top_three),
                    }
                )

        average_length = (
            sum(len(chunk["text"]) for chunk in chunks) / len(chunks) if chunks else 0
        )

        total = len(questions)

        summary = {
            "collection": name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": collection.count(),
            "avg_chunk_length": round(average_length, 2),
            "hit_at_1": round(hits[1] / total, 4),
            "hit_at_3": round(hits[3] / total, 4),
            "hit_at_5": round(hits[5] / total, 4),
        }

        summaries.append(summary)

        save_failures(
            RESULTS_DIR / f"failures_{chunk_size}_{overlap}.csv",
            failures,
        )

        print(f"Chunks: {summary['num_chunks']}")
        print(f"Average length: {summary['avg_chunk_length']}")
        print(f"Hit@1: {summary['hit_at_1']:.2%}")
        print(f"Hit@3: {summary['hit_at_3']:.2%}")
        print(f"Hit@5: {summary['hit_at_5']:.2%}")
        print(f"Failures at k=5: {len(failures)}")

    summary_path = RESULTS_DIR / "experiment_summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
