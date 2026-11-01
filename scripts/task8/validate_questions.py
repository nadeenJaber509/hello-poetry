"""Validate evaluation questions against the documentation."""

import csv
import sys
from pathlib import Path

from hello_poetry.documents import load_documents

QUESTIONS_PATH = Path("data/questions.csv")
DOCS_PATH = Path("data/poetry_docs")


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    documents = {doc["source"]: doc for doc in load_documents(DOCS_PATH)}

    with QUESTIONS_PATH.open(encoding="utf-8", newline="") as file:
        questions = list(csv.DictReader(file))

    print(f"Total questions: {len(questions)}")

    for question in questions:
        question_id = question["id"]
        source = question["expected_source"].strip()
        key_phrase = question["key_phrase"].strip()

        if not source:
            status = "NO ANSWER" if not key_phrase else "INVALID: unexpected key phrase"
        elif source not in documents:
            status = "INVALID: source not found"
        elif not key_phrase:
            status = "INVALID: missing key phrase"
        elif key_phrase.lower() not in documents[source]["text"].lower():
            status = "INVALID: key phrase not found"
        else:
            status = "SOURCE AND PHRASE FOUND"

        print(f"{question_id}: {status}")


if __name__ == "__main__":
    main()
