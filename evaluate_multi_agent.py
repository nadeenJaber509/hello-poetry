
import csv
import json
from pathlib import Path

from coordinator import run_support_team

CASES_FILE = Path("data/cases.csv")
RESULTS_FILE = Path("data/multi_agent_reviewer_on_detailed_results.csv")

FIELDNAMES = [
    "id",
    "type",
    "question",
    "status",
    "answerable",
    "answer",
    "commands",
    "sources",
    "confidence",
    "route",
    "research",
    "draft",
    "model_calls",
    "max_model_calls",
    "budget_exhausted",
    "review_decision",
    "review_count",
    "reviewer_feedback",
    "expected_source",
    "key_phrase",
    "phrase_match",
    "source_match",
    "error",
]


def to_json(value):
    """Convert objects into JSON-safe strings."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def load_completed_ids():
    """Resume without repeating successfully evaluated cases."""
    if not RESULTS_FILE.exists():
        return set()

    with RESULTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return {
            row["id"]
            for row in csv.DictReader(file)
            if row.get("status") == "success"
        }


def evaluate_case(case):
    """Run the multi-agent team on one case."""
    result = run_support_team(
    case["question"],
    enable_reviewer=True,
)

    answer = result.get("answer")

    if answer is not None and hasattr(answer, "model_dump"):
        answer = answer.model_dump()

    if answer is None:
        status = "incomplete"
        answer_text = ""
        answerable = ""
        commands = []
        sources = []
        confidence = ""
    else:
        status = "success"
        answer_text = answer.get("answer", "")
        answerable = answer.get("answerable", "")
        commands = answer.get("commands", [])
        sources = answer.get("sources", [])
        confidence = answer.get("confidence", "")

    expected_phrase = case.get("key_phrase", "").strip()
    expected_source = case.get("expected_source", "").strip()

    phrase_match = (
        bool(expected_phrase)
        and expected_phrase.lower() in answer_text.lower()
    )

    sources_text = to_json(sources).lower()
    source_match = (
        bool(expected_source)
        and expected_source.lower() in sources_text
    )

    return {
        "id": case["id"],
        "type": case.get("type", ""),
        "question": case["question"],
        "status": status,
        "answerable": answerable,
        "answer": answer_text,
        "commands": to_json(commands),
        "sources": to_json(sources),
        "confidence": confidence,
        "route": result.get("route"),
        "research": to_json(result.get("research")),
        "draft": to_json(result.get("draft")),
        "model_calls": result.get("model_calls", 0),
        "max_model_calls": result.get("max_model_calls", 12),
        "budget_exhausted": result.get("budget_exhausted", False),
        "review_decision": result.get("review_decision", ""),
        "review_count": result.get("review_count", 0),
        "reviewer_feedback": result.get("reviewer_feedback", ""),
        "expected_source": expected_source,
        "key_phrase": expected_phrase,
        "phrase_match": phrase_match,
        "source_match": source_match,
        "error": result.get("error", ""),
    }


def main():
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"Cases file not found: {CASES_FILE}"
        )

    with CASES_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        cases = list(csv.DictReader(file))
        cases = cases[:10]

    completed_ids = load_completed_ids()

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()
        processed_this_run = 0
        max_cases_per_run = 3
        for index, case in enumerate(cases, start=1):

            case_id = case["id"]

            if case_id in completed_ids:
                print(
                    f"[{index}/{len(cases)}] "
                    f"Skipping completed case {case_id}"
                )
                continue

            print(
                f"\n[{index}/{len(cases)}] "
                f"Evaluating case {case_id}..."
            )

            try:
                row = evaluate_case(case)

            except (KeyError, TypeError, ValueError) as error:
                row = {
                    "id": case_id,
                    "type": case.get("type", ""),
                    "question": case.get("question", ""),
                    "status": "error",
                    "error": str(error),
                }

            writer.writerow(row)
            output_file.flush()
            processed_this_run += 1
            print(
                f"Status: {row['status']} | "
                f"Model calls: {row.get('model_calls', 0)}"
            )

            # Stop immediately on a Groq rate-limit error.
            if "rate limit" in str(row.get("error", "")).lower():
                print(
                    "\nGroq rate limit reached. "
                    "Evaluation stopped to avoid more failed calls."
                )
                break
            if processed_this_run >= max_cases_per_run:
                print(
                    "\nBatch limit reached. "
                    "Run the script again to continue."
                )
                break
    print("\nEvaluation finished.")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
