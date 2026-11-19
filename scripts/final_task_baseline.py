
"""Run the Part 3.1 baseline without guardrails."""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from groq import GroqError

from hello_poetry.agent import run_agent
from hello_poetry.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "data" / "cases.csv"
OUTPUT_FILE = (
    ROOT / "results" / "final_task" / "before_guardrails.csv"
)

DEFAULT_MODEL = "openai/gpt-oss-20b"

# Baseline prompt: no scope, citation, or command guardrails.
SYSTEM_PROMPT = (
    "You are a helpful support assistant for Python Poetry. "
    "Use the available documentation tools when helpful. "
    "Answer the user's question clearly."
)

RESULT_FIELDS = [
    "id",
    "type",
    "question",
    "expected_source",
    "key_phrase",
    "should_refuse",
    "final_answer",
    "model_calls",
    "total_tokens",
    "total_seconds",
    "stopped_by_limit",
    "steps",
    "trace_path",
    "status",
    "error",
]


def read_csv(path):
    """Read rows from a CSV file."""
    if not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def save_results(rows, output=OUTPUT_FILE):
    """Atomically save all completed and failed results."""
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output.parent,
            prefix="baseline_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)

            writer = csv.DictWriter(
                file,
                fieldnames=RESULT_FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, output)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def evaluate_case(case, model, max_steps):
    """Run one question through the unguarded agent."""
    result = run_agent(
        question=case["question"],
        system_prompt=SYSTEM_PROMPT,
        tool_schemas=TOOL_SCHEMAS,
        tool_functions=TOOL_FUNCTIONS,
        model=model,
        max_steps=max_steps,
    )

    return {
        **{field: case[field] for field in [
            "id",
            "type",
            "question",
            "expected_source",
            "key_phrase",
            "should_refuse",
        ]},
        "final_answer": result["final_answer"] or "",
        "model_calls": result["model_calls"],
        "total_tokens": result["total_tokens"],
        "total_seconds": result["total_seconds"],
        "stopped_by_limit": result["stopped_by_limit"],
        "steps": json.dumps(
            result["steps"],
            ensure_ascii=False,
        ),
        "trace_path": result["trace_path"],
        "status": "completed" if result["final_answer"] else "empty",
        "error": "",
    }


def main():
    """Run pending cases and resume from saved results."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum new cases to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--sample-ten", action="store_true")

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.sample_ten and (args.model != DEFAULT_MODEL or args.max_steps != 5):
        parser.error("The saved sample baseline uses the default model and five steps")

    cases = read_csv(CASES_FILE)

    if len(cases) != 40:
        raise ValueError(
            f"Expected 40 cases, found {len(cases)}"
        )

    output = (
        OUTPUT_FILE.with_name("before_guardrails_sample10.csv")
        if args.sample_ten
        else OUTPUT_FILE
    )
    if args.sample_ten:
        from scripts.final_task_repeat import selected_cases

        cases = selected_cases(CASES_FILE)
    existing = read_csv(output)
    if args.sample_ten and not existing:
        original = {row["id"]: row for row in read_csv(OUTPUT_FILE)}
        for case in cases:
            saved = original.get(case["id"])
            if saved and saved["status"] == "completed":
                if saved["question"] != case["question"]:
                    raise ValueError(f"Changed baseline question for {case['id']}")
                existing.append(
                    {
                        **saved,
                        "status": "completed" if saved["final_answer"].strip() else "empty",
                    }
                )
        save_results(existing, output)

    completed = {
        row["id"]
        for row in existing
        if row["status"] in {"completed", "empty"}
    }

    results_by_id = {
        row["id"]: row
        for row in existing
    }

    pending = [
        case
        for case in cases
        if case["id"] not in completed
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Total cases: {len(cases)}")
    print(f"Already completed: {len(completed)}")
    print(f"Running now: {len(pending)}")

    for index, case in enumerate(pending, start=1):
        case_id = case["id"]

        print(
            f"\n[{index}/{len(pending)}] "
            f"{case_id}: {case['question']}",
            flush=True,
        )

        try:
            result = evaluate_case(
                case,
                args.model,
                args.max_steps,
            )

        except (GroqError, OSError) as exc:
            # Save failure; a subsequent run can retry it.
            result = {
                **{field: case[field] for field in [
                    "id",
                    "type",
                    "question",
                    "expected_source",
                    "key_phrase",
                    "should_refuse",
                ]},
                "final_answer": "",
                "model_calls": "",
                "total_tokens": "",
                "total_seconds": "",
                "stopped_by_limit": "",
                "steps": "[]",
                "trace_path": "",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        results_by_id[case_id] = result

        # Save immediately after every question.
        ordered_results = [
            results_by_id[row["id"]]
            for row in cases
            if row["id"] in results_by_id
        ]

        save_results(ordered_results, output)

        print(
            f"Status: {result['status']}"
            f" | Calls: {result['model_calls']}"
            f" | Tokens: {result['total_tokens']}"
            f" | Seconds: {result['total_seconds']}"
        )

        if result["status"] == "completed":
            print(
                "Answer:",
                str(result["final_answer"])[:300],
            )
        else:
            print("Error:", result["error"])
            if "429" in result["error"] and any(
                term in result["error"].lower()
                for term in ("daily", "per day", "tpd", "rpd")
            ):
                break

    final_results = read_csv(output)
    counts = Counter(
        row["status"] for row in final_results
    )

    print("\n=== BASELINE SUMMARY ===")
    print("Completed:", counts["completed"])
    print("Errors:", counts["error"])
    print("Remaining:", len(cases) - counts["completed"] - counts["empty"])
    print("Saved:", output)


if __name__ == "__main__":
    main()
