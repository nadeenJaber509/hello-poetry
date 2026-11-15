
"""Prepare candidate Poetry questions for manual source verification."""

import csv
from pathlib import Path

from hello_poetry.tools import search_docs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "normal_candidates.csv"

QUESTIONS = [
    "How do I initialize Poetry in an existing project?",
    "How do I create a new project using Poetry?",
    "What information is stored in pyproject.toml?",
    "How do I remove a dependency from my project?",
    "How do I add a dependency to a specific group?",
    "How can I specify a dependency version constraint?",
    "How can I use dependencies from Git repositories?",
    "How do I check information about Poetry's virtual environment?",
    "How do I select a Python version for my Poetry environment?",
    "How do I run a Python script using Poetry?",
    "What happens when I run poetry install?",
    "How can I synchronize installed dependencies?",
    "What is the purpose of poetry.lock?",
    "How can I list installed project dependencies?",
    "How do I update project dependencies?",
    "How can I validate my pyproject.toml file?",
    "How can I change Poetry configuration settings?",
    "How do I build a distribution package with Poetry?",
    "How do I publish a package using Poetry?",
    "How do I configure a custom package repository?",
]


def main():
    """Search documentation and save candidates for review."""
    if OUTPUT.exists():
        raise FileExistsError(
            "normal_candidates.csv already exists. "
            "Review it instead of overwriting it."
        )

    rows = []

    for index, question in enumerate(QUESTIONS, start=11):
        print(f"Checking Q{index}: {question}")

        results = search_docs(question, k=3)

        best = results[0] if results else None

        rows.append({
            "id": f"Q{index:02d}",
            "question": question,
            "suggested_source": best["source"] if best else "",
            "similarity": (
                round(best["similarity"], 4)
                if best else ""
            ),
            "evidence_preview": (
                best["text"][:350].replace("\n", " ")
                if best else ""
            ),
            "expected_source": "",
            "key_phrase": "",
            "verified": False,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "suggested_source",
                "similarity",
                "evidence_preview",
                "expected_source",
                "key_phrase",
                "verified",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} candidates to {OUTPUT}")
    print("Review the sources and fill the verification fields.")


if __name__ == "__main__":
    main()
