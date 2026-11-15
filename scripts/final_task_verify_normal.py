
"""Attach evidence-backed labels to normal evaluation candidates."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "poetry_docs"
INPUT_FILE = ROOT / "data" / "normal_candidates.csv"
OUTPUT_FILE = ROOT / "data" / "normal_review.csv"

EVIDENCE = {
    "Q11": ("basic-usage.md", "poetry init"),
    "Q12": ("basic-usage.md", "poetry new"),
    "Q13": ("pyproject.md", "pyproject.toml"),
    "Q14": ("cli.md", "poetry remove"),
    "Q15": ("managing-dependencies.md", "tool.poetry.group"),
    "Q16": ("dependency-specification.md", "~=1.2"),
    "Q17": ("dependency-specification.md", "git+"),
    "Q18": ("managing-environments.md", "poetry env info"),
    "Q19": ("managing-environments.md", "poetry env use"),
    "Q20": ("basic-usage.md", "poetry run python"),
    "Q21": ("cli.md", "poetry install"),
    "Q22": ("cli.md", "poetry sync"),
    "Q23": ("basic-usage.md", "poetry.lock"),
    "Q24": ("cli.md", "poetry show"),
    "Q25": ("cli.md", "poetry update"),
    "Q26": ("cli.md", "poetry check"),
    "Q27": ("configuration.md", "poetry config"),
    "Q28": ("cli.md", "poetry build"),
    "Q29": ("cli.md", "poetry publish"),
    "Q30": ("repositories.md", "poetry source add"),
}

REVISED_QUESTIONS = {
    "Q15": "How are dependency groups declared in Poetry?",
    "Q30": (
        "How do I add a custom package source "
        "for installing dependencies in Poetry?"
    ),
}


def main():
    """Create a review file with evidence checks."""
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "normal_review.csv already exists. "
            "Review it before creating a new version."
        )

    df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

    for column in ["expected_source", "key_phrase"]:
        if column not in df.columns:
            df[column] = ""

    df["evidence_found"] = False
    df["verified"] = False

    for index, row in df.iterrows():
        question_id = row["id"]

        if question_id not in EVIDENCE:
            continue

        source, phrase = EVIDENCE[question_id]

        if question_id in REVISED_QUESTIONS:
            df.at[index, "question"] = REVISED_QUESTIONS[question_id]

        path = DOCS_DIR / source

        if not path.is_file():
            print(f"{question_id}: Missing source {source}")
            continue

        document = path.read_text(encoding="utf-8")

        found = phrase.lower() in document.lower()

        df.at[index, "expected_source"] = source
        df.at[index, "key_phrase"] = phrase
        df.at[index, "evidence_found"] = found

        print(
            f"{question_id}: "
            f"{'FOUND' if found else 'NOT FOUND'} "
            f"| {source} | {phrase}"
        )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Evidence found:", int(df["evidence_found"].sum()))
    print("Manually verified:", int(df["verified"].sum()))
    print("Total candidates:", len(df))


if __name__ == "__main__":
    main()
