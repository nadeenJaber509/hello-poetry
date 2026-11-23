
import csv
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "poetry-support-team-evaluation"
CSV_PATH = Path("data/cases.csv")


def main():
    client = Client()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = list(client.list_examples(dataset_id=dataset.id))

        print(f"Dataset already exists: {DATASET_NAME}")
        print(f"Existing examples: {len(existing)}")
        print("No examples uploaded to avoid duplicates.")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "40 evaluation cases for the Poetry Support Team, "
            "including normal, special, and prompt-injection cases."
        ),
    )

    examples = []

    for row in rows:
        examples.append({
            "inputs": {
                "question": row["question"],
            },
            "reference_outputs": {
                "key_phrase": row["key_phrase"],
                "expected_source": row["expected_source"],
                "answerable": row["answerable"].strip().lower() == "yes",
            },
            "metadata": {
                "case_id": row["id"],
                "type": row["type"],
                "difficulty": row["difficulty"],
            },
        })

    client.create_examples(
        dataset_name=DATASET_NAME,
        examples=examples,
    )

    print(f"Dataset created: {DATASET_NAME}")
    print(f"Examples uploaded: {len(examples)}")


if __name__ == "__main__":
    main()
