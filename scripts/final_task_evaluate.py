"""Checkpointed final-task evaluation with explicit case limits and retries."""

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

from groq import APIConnectionError, APIStatusError, APITimeoutError

from hello_poetry.coordinator import answer_with_team
from hello_poetry.guardrails import REFUSAL, guarded_answer
from scripts.task9.run_rag_eval import COLLECTION_NAME, MODEL_NAME, PLAIN_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "cases.csv"
RESULTS = ROOT / "results" / "final_task"
FIELDS = (
    "id",
    "type",
    "question",
    "answer",
    "refused",
    "sources",
    "json_ok_first_try",
    "model_calls",
    "tokens",
    "seconds",
    "status",
    "error",
    "blocked_by",
    "commands",
    "trace_path",
    "citations_valid",
)


def explicit_doc_refusal(answer: str) -> bool:
    """Conservatively detect an explicit documentation-based refusal."""
    text = " ".join(answer.casefold().split())
    return any(
        phrase in text
        for phrase in (
            "cannot answer from the documents",
            "can't answer from the documents",
            "not found in the documentation",
            "no relevant documentation was found",
            "not in the provided documentation",
        )
    )


def _run_case(case, system):
    question = case["question"]
    if system == "plain":
        from hello_poetry.llm import ask_llm_with_usage

        answer, tokens = ask_llm_with_usage(
            PLAIN_SYSTEM_PROMPT, question, MODEL_NAME, temperature=0
        )
        return {
            "answer": answer,
            "sources": [],
            "model_calls": 1,
            "tokens": tokens,
            "refused": explicit_doc_refusal(answer),
            "citations_valid": None,
        }
    if system == "rag":
        from hello_poetry.rag import answer_question

        result = answer_question(question, COLLECTION_NAME, k=3, include_usage=True)
        answer = result["answer"]
        references = [int(number) for number in re.findall(r"\[(\d+)\]", answer)]
        return {
            "answer": answer,
            "sources": [s["source"] for s in result["sources"]],
            "model_calls": result.get("model_calls", 0),
            "tokens": result.get("tokens", 0),
            "refused": explicit_doc_refusal(answer),
            "citations_valid": bool(references)
            and all(1 <= number <= len(result["sources"]) for number in references),
        }
    if system == "single":
        result = guarded_answer(question)
    else:
        result = answer_with_team(question, use_reviewer=system == "team")
    return {
        **result,
        "refused": not result["answerable"] or result["answer"] == REFUSAL,
        "citations_valid": True if result["answerable"] else None,
    }


def _retry(case, system):
    from groq import RateLimitError

    for attempt in range(3):
        try:
            return _run_case(case, system)
        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIStatusError,
        ) as exc:
            if isinstance(exc, (APIConnectionError, APITimeoutError)):
                raise
            if isinstance(exc, APIStatusError) and exc.status_code == 429:
                raise
            if isinstance(exc, APIStatusError) and exc.status_code not in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
            }:
                raise
            if attempt == 2:
                raise
            time.sleep(min(8, 2**attempt))
    raise RuntimeError("Retry limit reached")


def run(system, limit=None, sample_ten=False, retry_id=None, retry_output=None):
    if system not in {"plain", "rag", "single", "team", "team_no_reviewer"}:
        raise ValueError("Unknown system")
    if sample_ten:
        from scripts.final_task_repeat import selected_cases

        cases = selected_cases(CASES)
    else:
        with CASES.open(newline="", encoding="utf-8-sig") as file:
            cases = list(csv.DictReader(file))
    suffix = "_sample10" if sample_ten else ""
    original_path = RESULTS / f"results_{system}{suffix}.csv"
    path = original_path
    if retry_id is not None:
        target = next((case for case in cases if case["id"] == retry_id), None)
        if target is None or not original_path.exists():
            raise ValueError(f"No saved case to retry: {retry_id}")
        with original_path.open(newline="", encoding="utf-8-sig") as file:
            original = {row["id"]: row for row in csv.DictReader(file)}.get(retry_id)
        if (
            original is None
            or original["question"] != target["question"]
            or original["status"] not in {"error", "refused"}
        ):
            raise ValueError(
                f"Only a saved error or refusal can be retried: {retry_id}"
            )
        cases = [target]
        path = (
            retry_output or RESULTS / f"results_{system}{suffix}_retry_{retry_id}.csv"
        )
        if path == original_path:
            raise ValueError("Retry output must differ from the original results")
    saved = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as file:
            saved = {row["id"]: row for row in csv.DictReader(file)}
        if retry_id is not None and set(saved) != {retry_id}:
            raise ValueError("Retry output contains a different case")
    for case in cases[:limit]:
        previous = saved.get(case["id"])
        if previous:
            if previous["question"] != case["question"]:
                raise ValueError(f"Changed question for {case['id']}")
            continue
        start = time.perf_counter()
        try:
            result = _retry(case, system)
            status = (
                "empty"
                if not result["answer"].strip()
                else ("refused" if result["refused"] else "completed")
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 - checkpoint every failure separately
            result = {}
            status, error = "error", f"{type(exc).__name__}: {exc}"
        row = {
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "answer": result.get("answer", ""),
            "refused": str(result.get("refused", False)).lower(),
            "sources": json.dumps(result.get("sources", [])),
            "json_ok_first_try": str(result.get("json_ok_first_try", "")).lower(),
            "model_calls": result.get("model_calls", ""),
            "tokens": result.get("total_tokens", result.get("tokens", "")),
            "seconds": result.get("total_seconds", time.perf_counter() - start),
            "status": status,
            "error": error,
            "blocked_by": result.get("blocked_by", ""),
            "commands": json.dumps(result.get("commands", [])),
            "trace_path": result.get("trace_path", ""),
            "citations_valid": str(result.get("citations_valid", "")).lower(),
        }
        saved[case["id"]] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(saved.values())
        temporary.replace(path)
        if status == "error":
            break
        time.sleep(0.5)
    if system == "single" and retry_id is None and len(saved) == len(cases):
        after = RESULTS / f"after_guardrails{suffix}.csv"
        if after.exists() and after.read_bytes() != path.read_bytes():
            raise FileExistsError(
                "Existing after_guardrails.csv differs; review it before exporting"
            )
        if not after.exists():
            shutil.copyfile(path, after)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=("plain", "rag", "single", "team", "team_no_reviewer"),
        required=True,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit", type=int)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--retry-id")
    parser.add_argument("--sample-ten", action="store_true")
    parser.add_argument("--retry-output", type=Path)
    args = parser.parse_args()
    maximum = 10 if args.sample_ten else 40
    if args.limit is not None and not 1 <= args.limit <= maximum:
        parser.error(f"--limit must be between 1 and {maximum}")
    if args.retry_output is not None and args.retry_id is None:
        parser.error("--retry-output requires --retry-id")
    if args.retry_id is not None and not args.sample_ten:
        parser.error("--retry-id requires --sample-ten")
    print(
        run(
            args.system,
            None if args.all else args.limit,
            args.sample_ten,
            retry_id=args.retry_id,
            retry_output=args.retry_output,
        )
    )
