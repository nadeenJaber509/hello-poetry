"""Explicit Part 8 graph comparison and LangSmith experiments."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from hello_poetry.agents import run_researcher, run_reviewer, run_router, run_writer
from hello_poetry.coordinator import answer_with_team
from hello_poetry.team_graph import answer_with_graph, build_graph
from scripts.final_task_evaluate import CASES, RESULTS
from scripts.final_task_repeat import selected_cases

DATASET_NAME = "poetry-support-team-40"
AGENTS = {
    "router": run_router,
    "researcher": run_researcher,
    "writer": run_writer,
    "reviewer": run_reviewer,
}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _retry(call, stop_on_429: bool = False):
    for attempt in range(3):
        try:
            return call()
        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIStatusError,
        ) as exc:
            if (
                stop_on_429
                and isinstance(exc, APIStatusError)
                and exc.status_code == 429
            ):
                raise
            if (
                isinstance(exc, APIStatusError)
                and exc.status_code == 429
                and any(
                    term in str(exc).lower()
                    for term in ("daily", "per day", "tpd", "rpd")
                )
            ):
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
            time.sleep(2**attempt)
    raise RuntimeError("Retry limit reached")


def compare_graph(
    limit: int = 10,
    output: Path = RESULTS / "graph_comparison.json",
    cases_path: Path = CASES,
    case_ids: list[str] | None = None,
) -> dict:
    """Checkpoint the same ten cases for coordinator and graph."""
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    build_graph()  # Check the optional package before making any model call.
    selected = selected_cases(cases_path)
    if case_ids is not None:
        by_id = {case["id"]: case for case in selected}
        if (
            not case_ids
            or len(case_ids) != len(set(case_ids))
            or any(case_id not in by_id for case_id in case_ids)
        ):
            raise ValueError("Choose unique case IDs from the fixed ten-case sample")
        selected = [by_id[case_id] for case_id in case_ids]
    else:
        selected = selected[:limit]
    rows = _load(output)
    for case in selected:
        for name, runner in (
            ("coordinator", answer_with_team),
            ("graph", answer_with_graph),
        ):
            key = f"{case['id']}:{name}"
            if key in rows:
                if rows[key]["question"] != case["question"]:
                    raise ValueError(f"Saved question changed for {key}")
                continue
            try:
                result = _retry(
                    lambda runner=runner, question=case["question"]: runner(question)
                )
                rows[key] = {
                    "question": case["question"],
                    "status": "completed",
                    "result": result,
                }
            except Exception as exc:  # noqa: BLE001 - keep failures separate
                rows[key] = {
                    "question": case["question"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _save(output, rows)
            if (
                rows[key]["status"] == "error"
                and "429" in rows[key]["error"]
                and any(
                    term in rows[key]["error"].lower()
                    for term in ("daily", "per day", "tpd", "rpd")
                )
            ):
                return rows
            time.sleep(0.5)
    return rows


def retry_failed_graph(
    case_id: str,
    comparison: Path = RESULTS / "graph_comparison.json",
    output: Path | None = None,
) -> dict:
    """Run only a failed graph case and save it separately from the comparison."""
    rows = _load(comparison)
    coordinator = rows.get(f"{case_id}:coordinator")
    graph = rows.get(f"{case_id}:graph")
    if not coordinator or coordinator.get("status") != "completed":
        raise ValueError(f"No completed coordinator case: {case_id}")
    if not graph or graph.get("status") != "error":
        raise ValueError(f"No failed graph case: {case_id}")
    if coordinator["question"] != graph["question"]:
        raise ValueError(f"Saved questions differ for {case_id}")
    output = output or RESULTS / f"graph_retry_{case_id}.json"
    if output == comparison:
        raise ValueError("Retry output must differ from the original comparison")
    if output.exists():
        saved = _load(output)
        if saved.get("question") != graph["question"]:
            raise ValueError(f"Saved retry question changed for {case_id}")
        return saved
    try:
        result = _retry(lambda: answer_with_graph(graph["question"]))
        saved = {"question": graph["question"], "status": "completed", "result": result}
    except Exception as exc:  # noqa: BLE001 - preserve failed retries
        saved = {
            "question": graph["question"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _save(output, saved)
    return saved


def graph_summary(rows: dict) -> dict:
    ids = sorted({key.split(":")[0] for key in rows})
    pairs = [
        (rows[f"{case_id}:coordinator"], rows[f"{case_id}:graph"])
        for case_id in ids
        if f"{case_id}:coordinator" in rows and f"{case_id}:graph" in rows
    ]
    usable = [(a, b) for a, b in pairs if a["status"] == b["status"] == "completed"]
    return {
        "pairs": len(pairs),
        "usable": len(usable),
        "same_answer": sum(
            a["result"]["answer"] == b["result"]["answer"] for a, b in usable
        ),
    }


def _langsmith():
    load_dotenv()
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is not configured")
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        raise ValueError("Set LANGSMITH_TRACING=true before sending traces")
    try:
        from langsmith import Client, evaluate, traceable
    except ImportError as exc:
        raise RuntimeError("Install langsmith for the Part 8 experiment") from exc
    return Client, evaluate, traceable


def traced_team(question: str) -> dict:
    """Trace the same four plain-Python agents without changing their logic."""
    _, _, traceable = _langsmith()
    traced = {name: traceable(name=name)(function) for name, function in AGENTS.items()}

    @traceable(name="poetry_support_team")
    def run():
        return answer_with_team(question, agent_functions=traced)

    return run()


def trace_ten(
    limit: int = 10,
    output: Path = RESULTS / "langsmith_traces.json",
    cases_path: Path = CASES,
    retry_errors: bool = False,
) -> dict:
    """Checkpoint ten public test cases traced to LangSmith."""
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    rows = _load(output)
    for case in selected_cases(cases_path)[:limit]:
        previous = rows.get(case["id"])
        if previous:
            if previous["question"] != case["question"]:
                raise ValueError(f"Saved question changed for {case['id']}")
            if previous["status"] != "error" or not retry_errors:
                continue
        _langsmith()
        history = (
            [
                *previous.get("history", []),
                {key: value for key, value in previous.items() if key != "history"},
            ]
            if previous
            else []
        )
        rate_limited = False
        try:
            result = _retry(
                lambda question=case["question"]: traced_team(question),
                stop_on_429=True,
            )
            rows[case["id"]] = {
                "question": case["question"],
                "status": "completed",
                "result": result,
            }
        except Exception as exc:  # noqa: BLE001 - preserve failed calls
            rows[case["id"]] = {
                "question": case["question"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rate_limited = isinstance(exc, APIStatusError) and exc.status_code == 429
        if history:
            rows[case["id"]]["history"] = history
        _save(output, rows)
        if rate_limited:
            return rows
        time.sleep(0.5)
    return rows


def upload_dataset(name: str = DATASET_NAME, cases_path: Path = CASES) -> str:
    """Upload exactly the public 40-case dataset, once, when explicitly invoked."""
    Client, _, _ = _langsmith()
    with cases_path.open(newline="", encoding="utf-8-sig") as file:
        cases = list(csv.DictReader(file))
    if len(cases) != 40 or len({case["id"] for case in cases}) != 40:
        raise ValueError("Expected 40 unique cases")
    client = Client()
    if client.has_dataset(dataset_name=name):
        raise FileExistsError(f"LangSmith dataset already exists: {name}")
    dataset = client.create_dataset(dataset_name=name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": case["question"]},
                "outputs": {
                    "id": case["id"],
                    "should_refuse": case["should_refuse"],
                    "expected_source": case["expected_source"],
                    "key_phrase": case["key_phrase"],
                },
            }
            for case in cases
        ],
    )
    return str(dataset.id)


def langsmith_experiment(
    name: str = DATASET_NAME, output: Path = RESULTS / "langsmith_experiment.json"
) -> dict:
    """Run one remote 40-case experiment only when explicitly invoked."""
    Client, evaluate, _ = _langsmith()
    if output.exists():
        raise FileExistsError(f"Saved experiment already exists: {output}")
    if not Client().has_dataset(dataset_name=name):
        raise ValueError(f"Upload the LangSmith dataset first: {name}")

    def target(inputs):
        result = _retry(lambda: traced_team(inputs["question"]))
        return {"answer": result["answer"], "answerable": result["answerable"]}

    experiment = evaluate(
        target, data=name, experiment_prefix="poetry-support-team", max_concurrency=1
    )
    record = {
        "dataset": name,
        "experiment_name": getattr(experiment, "experiment_name", None),
        "status": "submitted; verify all 40 runs in LangSmith",
    }
    _save(output, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--compare-graph", action="store_true")
    actions.add_argument("--retry-graph-case", metavar="CASE_ID")
    actions.add_argument("--graph-summary", action="store_true")
    actions.add_argument("--trace-ten", action="store_true")
    actions.add_argument("--upload-dataset", action="store_true")
    actions.add_argument("--langsmith-experiment", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.retry_errors and not args.trace_ten:
        parser.error("--retry-errors requires --trace-ten")
    if args.compare_graph:
        print(
            graph_summary(
                compare_graph(
                    args.limit,
                    output=args.output or RESULTS / "graph_comparison.json",
                    case_ids=args.case_id,
                )
            )
        )
    elif args.retry_graph_case:
        retry_path = (
            args.output or RESULTS / f"graph_retry_{args.retry_graph_case}.json"
        )
        result = retry_failed_graph(args.retry_graph_case, output=retry_path)
        print(f"Saved {result['status']} graph retry: {retry_path}")
    elif args.graph_summary:
        print(graph_summary(_load(RESULTS / "graph_comparison.json")))
    elif args.trace_ten:
        print(
            f"Saved {len(trace_ten(args.limit, retry_errors=args.retry_errors))} traced cases"
        )
    elif args.upload_dataset:
        print(f"Uploaded dataset: {upload_dataset()}")
    else:
        print(langsmith_experiment())
