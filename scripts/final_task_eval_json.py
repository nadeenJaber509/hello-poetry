
"""Evaluate Pydantic validation with and without Groq JSON mode."""

import argparse
import json
from pathlib import Path

import pandas as pd
from groq import GroqError

from hello_poetry.schemas import get_structured_answer

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = ROOT / "data" / "questions.csv"
RESULTS_DIR = ROOT / "results" / "final_task"

MODEL = "openai/gpt-oss-20b"
MAX_RETRIES = 2


def save_results(path, results):
    """Save results after each completed question."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate(mode):
    """Run the JSON validation experiment."""
    json_mode = mode == "json"
    output_file = RESULTS_DIR / f"json_eval_{mode}.json"

    questions = pd.read_csv(
        QUESTIONS_FILE,
        dtype=str,
    ).fillna("")

    required_columns = {"id", "question"}

    if not required_columns.issubset(questions.columns):
        raise ValueError(
            "The dataset must include id and question columns."
        )

    if questions["id"].duplicated().any():
        raise ValueError("Question IDs must be unique.")

    if questions["question"].str.strip().eq("").any():
        raise ValueError("Questions must not be empty.")

    if output_file.exists():
        results = json.loads(
            output_file.read_text(encoding="utf-8")
        )
    else:
        results = {}

    print(f"\nMode: {mode}")
    print(f"Model: {MODEL}")
    print(f"Questions: {len(questions)}")

    for _, row in questions.iterrows():
        question_id = str(row["id"])
        question = str(row["question"])

        if question_id in results:
            saved = results[question_id]

            if saved["question"] != question:
                raise ValueError(
                    f"Question {question_id} has changed."
                )

            if saved["status"] != "api_error":
                print(f"SKIPPING {question_id}: already evaluated")
                continue

            print(
                f"STOPPED at {question_id}: previous API error. "
                "Review the saved response before retrying."
            )
            break

        print(f"\nEvaluating {question_id}: {question}")

        messages = [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    "Return a structured answer. "
                    "Use only evidence provided in this conversation. "
                    "If no documentation evidence is provided, "
                    "do not invent sources or commands."
                ),
            },
        ]

        try:
            answer, retries = get_structured_answer(
                messages=messages,
                model=MODEL,
                max_retries=MAX_RETRIES,
                json_mode=json_mode,
            )

        except ValueError as error:
            # Validation failed after the allowed attempts.
            results[question_id] = {
                "question": question,
                "status": "validation_failed",
                "valid_first_try": False,
                "retries": MAX_RETRIES,
                "model_calls": MAX_RETRIES + 1,
                "error": str(error),
            }

            print(f"VALIDATION FAILED: {error}")

        except (GroqError, OSError) as error:
            # Stop instead of automatically repeating API calls.
            results[question_id] = {
                "question": question,
                "status": "api_error",
                "error": str(error),
            }

            print(f"API ERROR: {error}")
            save_results(output_file, results)
            break

        else:
            results[question_id] = {
                "question": question,
                "status": "valid",
                "valid_first_try": retries == 0,
                "retries": retries,
                "model_calls": retries + 1,
                "answer": answer.model_dump(),
            }

            print(
                f"VALID | Retries: {retries} | "
                f"First try: {retries == 0}"
            )

        save_results(output_file, results)

    # ------------------------------------------
    # Summary
    # ------------------------------------------

    records = list(results.values())

    valid = [
        record
        for record in records
        if record["status"] == "valid"
    ]

    first_try = sum(
        record["valid_first_try"]
        for record in valid
    )

    after_retry = sum(
        record["retries"] > 0
        for record in valid
    )

    failed = sum(
        record["status"] == "validation_failed"
        for record in records
    )

    api_errors = sum(
        record["status"] == "api_error"
        for record in records
    )

    total_calls = sum(
        record.get("model_calls", 0)
        for record in records
    )

    print("\n" + "=" * 40)
    print("JSON EVALUATION SUMMARY")
    print("=" * 40)

    print("Mode:", mode)
    print("Questions saved:", len(records))
    print("Valid first try:", first_try)
    print("Valid after retry:", after_retry)
    print("Validation failed:", failed)
    print("API errors:", api_errors)
    print("Recorded model calls:", total_calls)
    print("Results saved to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate structured JSON output."
    )

    parser.add_argument(
        "--mode",
        choices=["json", "prompt"],
        required=True,
    )

    args = parser.parse_args()
    evaluate(args.mode)
