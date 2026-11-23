
import csv
from pathlib import Path

DATA_DIR = Path("data")

SYSTEM_FILES = {
    "Plain LLM": Path("answers_plain_k3.csv"),
    "RAG": Path("answers_rag_k3.csv"),
    "Single Agent": DATA_DIR / "single_agent_results.csv",
    "Multi-Agent": DATA_DIR / "multi_agent_results.csv",
}

SUMMARY_FILE = DATA_DIR / "comparison_summary.csv"
COMMON_FILE = DATA_DIR / "common_case_comparison.csv"


def read_csv(path):
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def is_normal(row, normal_ids):
    case_id = str(row.get("id", "")).strip()

    if normal_ids:
        return case_id in normal_ids

    case_type = (
        row.get("type")
        or row.get("category")
        or ""
    ).strip().lower()

    return case_type == "normal"


def analyze_system(name, path, normal_ids, cases):
    rows = read_csv(path)

    if not rows:
        return {
            "system": name,
            "total_normal": len(normal_ids),
            "completed": 0,
            "incomplete": 0,
            "errors": 0,
            "phrase_matches": 0,
            "phrase_accuracy": "",
            "coverage": "0.0%",
            "results": {},
        }


    # Keep only the latest result for each case ID.
    latest_rows = {}

    for row in rows:
        case_id = str(row.get("id", "")).strip()

        if case_id in normal_ids:
            latest_rows[case_id] = row

    results = {}
    completed = 0
    incomplete = 0
    errors = 0
    phrase_matches = 0

    for row in latest_rows.values():

        case_id = str(row.get("id", "")).strip()

        if case_id not in normal_ids:
            continue

        status = row.get("status", "").strip().lower()

        # Plain LLM and RAG files have no status column.
        if not status:
            status = "success" if row.get("answer", "").strip() else "incomplete"

        if status == "error":
            errors += 1
            continue

        answer = row.get("answer", "").strip()

        if status in {"incomplete", "failed"} or not answer:
            incomplete += 1
            continue

        completed += 1

        key_phrase = (
            row.get("key_phrase")
            or cases.get(case_id, {}).get("key_phrase", "")
        ).strip()

        matched = (
            bool(key_phrase)
            and key_phrase.lower() in answer.lower()
        )

        if matched:
            phrase_matches += 1

        results[case_id] = {
            "answer": answer,
            "phrase_match": matched,
        }

    accuracy = (
        round(phrase_matches / completed * 100, 2)
        if completed else ""
    )

    coverage = (
        round(completed / len(normal_ids) * 100, 2)
        if normal_ids else 0
    )

    return {
        "system": name,
        "total_normal": len(normal_ids),
        "completed": completed,
        "incomplete": incomplete,
        "errors": errors,
        "phrase_matches": phrase_matches,
        "phrase_accuracy": accuracy,
        "coverage": f"{coverage}%",
        "results": results,
    }

def save_summary(summaries):
    fields = [
        "system",
        "total_normal",
        "completed",
        "incomplete",
        "errors",
        "phrase_matches",
        "phrase_accuracy",
        "coverage",
    ]

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()

        for summary in summaries:
            writer.writerow({
                field: summary[field]
                for field in fields
            })


def save_common_comparison(summaries):
    available = {
        summary["system"]: summary["results"]
        for summary in summaries
        if summary["results"]
    }

    system_names = list(available)

    if len(system_names) < 2:
        print("\nNot enough systems with results for comparison.")
        return

    common_ids = set.intersection(
        *[
            set(available[name])
            for name in system_names
        ]
    )

    fields = ["id"] + [
        f"{name}_match"
        for name in system_names
    ]

    with COMMON_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()

        for case_id in sorted(
            common_ids,
            key=lambda value: (
                not value.isdigit(),
                int(value) if value.isdigit() else value,
            ),
        ):
            row = {"id": case_id}

            for name in system_names:
                row[f"{name}_match"] = (
                    available[name][case_id]["phrase_match"]
                )

            writer.writerow(row)

    print(f"\nSystems compared: {', '.join(system_names)}")
    print(f"Common completed cases: {len(common_ids)}")
    print(f"Saved: {COMMON_FILE}")

def main():
    cases_rows = read_csv(DATA_DIR / "cases.csv")

    cases = {
        str(row["id"]).strip(): row
        for row in cases_rows
        if row.get("id")
    }

    normal_ids = {
        case_id
        for case_id, case in cases.items()
        if case.get("type", "").strip().lower() == "normal"
    }

    if not normal_ids:
        print("No normal cases found in data/cases.csv.")
        return

    summaries = []

    print("\n========== SYSTEM COMPARISON ==========\n")

    for name, path in SYSTEM_FILES.items():
        summary = analyze_system(
            name,
            path,
            normal_ids,
            cases,
        )

        summaries.append(summary)

        print(f"System: {name}")
        print(f"Completed: {summary['completed']}/{summary['total_normal']}")
        print(f"Phrase matches: {summary['phrase_matches']}")
        print(f"Phrase accuracy: {summary['phrase_accuracy']}")
        print(f"Coverage: {summary['coverage']}")
        print(f"Errors: {summary['errors']}")
        print(f"Incomplete: {summary['incomplete']}")
        print("---------------------------------------")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_summary(summaries)
    save_common_comparison(summaries)

    print("\nAnalysis completed.")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
