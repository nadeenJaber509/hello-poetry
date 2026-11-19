"""Repeat the same ten cases twice to inspect temperature-zero variation."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

from scripts.final_task_evaluate import CASES, MODEL_NAME, RESULTS, _retry

SEED = 42
NORMAL_SOURCES = (
    "cli.md",
    "basic-usage.md",
    "dependency-specification.md",
    "managing-environments.md",
    "configuration.md",
    "repositories.md",
)
SETTINGS = {
    "model": MODEL_NAME,
    "temperature": 0,
    "rag_k": 3,
    "team_max_model_calls": 12,
    "team_max_review_rounds": 2,
}
FIELDS = (
    "id",
    "repeat",
    "type",
    "question",
    "settings",
    "status",
    "answer",
    "refused",
    "sources",
    "model_calls",
    "tokens",
    "seconds",
    "error",
)
SYSTEMS = ("plain", "rag", "single", "team", "team_no_reviewer")


def selected_cases(path: Path = CASES) -> list[dict]:
    """Draw six normal, two special, and two injection cases with seed 42."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        cases = list(csv.DictReader(file))
    if len(cases) != 40 or len({case["id"] for case in cases}) != 40:
        raise ValueError("The repeat experiment requires 40 unique cases")
    rng = random.Random(SEED)
    selected = []
    for source in NORMAL_SOURCES:
        group = sorted(
            (
                case
                for case in cases
                if case["type"] == "normal" and case["expected_source"] == source
            ),
            key=lambda case: case["id"],
        )
        if not group:
            raise ValueError(f"No normal case for {source}")
        selected.append(rng.choice(group))
    for case_type in ("special", "injection"):
        group = sorted(
            (case for case in cases if case["type"] == case_type),
            key=lambda case: case["id"],
        )
        if len(group) < 2:
            raise ValueError(f"Not enough {case_type} cases for the fixed sample")
        selected.extend(rng.sample(group, 2))
    return selected


def load_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Unexpected repeat result columns: {path}")
        rows = list(reader)
    keys = [(row["id"], row["repeat"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate saved repeat rows")
    return dict(zip(keys, rows, strict=True))


def save_rows(path: Path, rows: dict[tuple[str, str], dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    temporary.replace(path)


def run(
    system: str,
    repeat: int,
    limit: int = 10,
    cases_path: Path = CASES,
    results_dir: Path = RESULTS,
) -> Path:
    """Checkpoint each result and never call again for a saved run."""
    if system not in SYSTEMS or repeat not in (1, 2) or not 1 <= limit <= 10:
        raise ValueError("Choose a valid system, repeat 1 or 2, and limit 1–10")
    cases = selected_cases(cases_path)[:limit]
    path = results_dir / f"repeat_{system}.csv"
    rows = load_rows(path)
    settings = json.dumps(SETTINGS, sort_keys=True)
    for case in cases:
        key = (case["id"], str(repeat))
        previous = rows.get(key)
        if previous:
            if (
                previous["question"] != case["question"]
                or previous["settings"] != settings
            ):
                raise ValueError(f"Saved case or settings changed for {case['id']}")
            continue
        if repeat == 2 and (case["id"], "1") not in rows:
            raise ValueError(f"Run repeat 1 before repeat 2 for {case['id']}")
        started = time.perf_counter()
        try:
            result = _retry(case, system)
            status = (
                "empty"
                if not result["answer"].strip()
                else "refused"
                if result["refused"]
                else "completed"
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 - save failed calls as errors
            result, status, error = {}, "error", f"{type(exc).__name__}: {exc}"
        rows[key] = {
            "id": case["id"],
            "repeat": str(repeat),
            "type": case["type"],
            "question": case["question"],
            "settings": settings,
            "status": status,
            "answer": result.get("answer", ""),
            "refused": str(result.get("refused", False)).lower(),
            "sources": json.dumps(result.get("sources", [])),
            "model_calls": result.get("model_calls", ""),
            "tokens": result.get("total_tokens", result.get("tokens", "")),
            "seconds": result.get("total_seconds", time.perf_counter() - started),
            "error": error,
        }
        save_rows(path, rows)
        time.sleep(0.5)
    return path


def compare(path: Path) -> dict:
    """Report observed answer variation without scoring correctness."""
    rows = load_rows(path)
    ids = sorted({case_id for case_id, _ in rows})
    paired = [
        (rows[(case_id, "1")], rows[(case_id, "2")])
        for case_id in ids
        if (case_id, "1") in rows and (case_id, "2") in rows
    ]
    usable = [
        (a, b)
        for a, b in paired
        if a["status"] in {"completed", "refused"}
        and b["status"] in {"completed", "refused"}
    ]
    changed = [a["id"] for a, b in usable if a["answer"].strip() != b["answer"].strip()]
    return {
        "paired": len(paired),
        "usable": len(usable),
        "different_answers": len(changed),
        "different_ids": changed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", type=int, choices=(1, 2))
    action.add_argument("--compare", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(RESULTS / f"repeat_{args.system}.csv"), indent=2))
    else:
        print(run(args.system, args.run, args.limit))
