"""Summarize only measured, successful final-task evaluation rows."""

import argparse
import csv
import random

from scripts.final_task_evaluate import CASES, RESULTS


def manual_grade_ids(cases, seed=42):
    """Choose the same 15 normal and all 10 refusal cases every time."""
    normal = sorted(
        case_id
        for case_id, row in cases.items()
        if row["should_refuse"].lower() == "false"
    )
    special = sorted(
        case_id
        for case_id, row in cases.items()
        if row["should_refuse"].lower() == "true"
    )
    return sorted(random.Random(seed).sample(normal, 15)) + special


def summarize(sample_ten=False):
    with CASES.open(newline="", encoding="utf-8-sig") as file:
        all_cases = {row["id"]: row for row in csv.DictReader(file)}
    if sample_ten:
        from scripts.final_task_repeat import selected_cases

        cases = {row["id"]: row for row in selected_cases(CASES)}
        print("Manual review all 10 sampled IDs:", ", ".join(cases))
    else:
        cases = all_cases
        print("Manual grade sample (seed 42):", ", ".join(manual_grade_ids(cases)))
    suffix = "_sample10" if sample_ten else ""
    for system in ("plain", "rag", "single", "team", "team_no_reviewer"):
        path = RESULTS / f"results_{system}{suffix}.csv"
        if not path.exists():
            print(f"{system}: no measured runs")
            continue
        with path.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        rows = [row for row in rows if row["id"] in cases]
        valid = [r for r in rows if r["status"] in {"completed", "refused"}]
        normals = [
            r for r in valid if cases[r["id"]]["should_refuse"].lower() == "false"
        ]
        special = [
            r for r in valid if cases[r["id"]]["should_refuse"].lower() == "true"
        ]
        phrase = sum(
            cases[r["id"]]["key_phrase"].casefold() in r["answer"].casefold()
            for r in normals
        )
        correct_refusals = sum(r["refused"] == "true" for r in special)
        false_refusals = sum(r["refused"] == "true" for r in normals)
        citation_rows = [
            r for r in valid if r.get("citations_valid") in {"true", "false"}
        ]
        citations = (
            f"{sum(r['citations_valid'] == 'true' for r in citation_rows)}/{len(citation_rows)}"
            if citation_rows
            else "n/a"
        )
        json_rows = [r for r in valid if r["json_ok_first_try"] in {"true", "false"}]
        json_first = (
            f"{sum(r['json_ok_first_try'] == 'true' for r in json_rows)}/{len(json_rows)}"
            if json_rows
            else "n/a"
        )

        def average(field, measured=valid):
            values = [float(r[field]) for r in measured if r[field]]
            return round(sum(values) / len(values), 2) if values else "n/a"

        print(
            f"{system}: {len(valid)}/{len(cases)} measured; errors={sum(r['status'] == 'error' for r in rows)}; "
            f"empty={sum(r['status'] == 'empty' for r in rows)}; "
            f"phrase={phrase}/{len(normals)}; correct_refusals={correct_refusals}/{len(special)}; "
            f"false_refusals={false_refusals}/{len(normals)}; citations={citations}; "
            f"json_first={json_first}; "
            f"calls={average('model_calls')}; tokens={average('tokens')}; seconds={average('seconds')}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ten", action="store_true")
    summarize(parser.parse_args().sample_ten)
