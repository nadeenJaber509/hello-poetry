
"""Test the scope classifier with real LLM responses."""

import argparse
import json
from pathlib import Path

from groq import GroqError

from hello_poetry.guardrails import check_scope

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final_task" / "scope_tests.json"

QUESTIONS = [
    ("How do I add a dependency using Poetry?", True),
    ("How can I configure a Poetry virtual environment?", True),
    ("What is the weather in Ramallah?", False),
    ("What is today's Bitcoin price?", False),
    ("What is my private PyPI password?", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    if args.limit < 1 or args.limit > len(QUESTIONS):
        parser.error("--limit must be between 1 and 5")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    results = []

    if OUTPUT.exists():
        results = json.loads(OUTPUT.read_text(encoding="utf-8"))

    completed = {row["question"] for row in results}

    for question, expected in QUESTIONS[:args.limit]:
        if question in completed:
            print(f"Skipping completed: {question}")
            continue

        print(f"\nQuestion: {question}")
        print(f"Expected in_scope: {expected}")

        try:
            decision = check_scope(
                question,
                model="openai/gpt-oss-20b",
            )
        except GroqError as exc:
            print(f"Groq API error: {exc}")
            break

        row = {
            "question": question,
            "expected_in_scope": expected,
            "actual_in_scope": decision.in_scope,
            "reason": decision.reason,
            "correct": decision.in_scope == expected,
        }
        results.append(row)

        OUTPUT.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Actual in_scope:", decision.in_scope)
        print("Reason:", decision.reason)
        print("Correct:", row["correct"])

    print(f"\nSaved results: {OUTPUT}")


if __name__ == "__main__":
    main()
