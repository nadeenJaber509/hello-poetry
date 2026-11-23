
import json
from pathlib import Path

import pandas as pd

from rag import answer_question
from store import build_vector_store
from task9_special_questions import QUESTIONS as SPECIAL_QUESTIONS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "baseline_results_fixed.csv"


def load_test_cases():
    cases = []

    # 30 normal questions
    normal_df = pd.read_csv(DATA_DIR / "questions.csv")

    for _, row in normal_df.iterrows():
        cases.append({
            "category": "normal",
            "question": row["question"],
            "expected_source": row.get("expected_source", ""),
            "key_phrase": row.get("key_phrase", ""),
        })

    # 7 special questions
    for question in SPECIAL_QUESTIONS:
        cases.append({
            "category": "special",
            "question": question,
            "expected_source": "",
            "key_phrase": "",
        })

    # 3 prompt injection questions
    injection_df = pd.read_csv(DATA_DIR / "injection_questions.csv")

    for _, row in injection_df.iterrows():
        cases.append({
            "category": "injection",
            "question": row["question"],
            "expected_source": "",
            "key_phrase": row.get("type", ""),
        })

    return cases


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH.name} already exists. "
            "Rename or back it up before starting a new run."
        )

    cases = load_test_cases()

    print(f"Total test cases: {len(cases)}")

    if len(cases) != 40:
        raise ValueError(
            f"Expected 40 test cases, found {len(cases)}."
        )

    print("\nBuilding vector store...")
    collection, embedding_model = build_vector_store()

    results = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]

        print(f"\n[{index}/40] {case['category']}")
        print(f"Question: {question}")

        try:
            result = answer_question(
                question=question,
                collection=collection,
                model=embedding_model,
                k=5,
            )

            record = {
                **case,
                "answer": result.get("answer", ""),
                "sources": json.dumps(
                    result.get("sources", []),
                    ensure_ascii=False,
                    default=str,
                ),
                "steps": json.dumps(
                    result.get("steps", []),
                    ensure_ascii=False,
                    default=str,
                ),
                "total_tokens": result.get("total_tokens", 0),
                "stopped_by_limit": result.get(
                    "stopped_by_limit", False
                ),
                "status": "success",
                "error": "",
            }
# Continue evaluating remaining questions if one case fails.

        except Exception as error: # noqa: BLE001
            record = {
                **case,
                "answer": "",
                "sources": "[]",
                "steps": "[]",
                "total_tokens": 0,
                "stopped_by_limit": False,
                "status": "error",
                "error": str(error),
            }

        results.append(record)

        # Save after every question
        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Status: {record['status']}")

    print("\nBaseline testing completed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(
        f"Successful: {sum(r['status'] == 'success' for r in results)}"
    )
    print(
        f"Failed: {sum(r['status'] == 'error' for r in results)}"
    )


if __name__ == "__main__":
    main()
