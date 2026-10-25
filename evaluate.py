
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from hello_poetry.llm import ask_llm, parse_label
from hello_poetry.prompts import PROMPT_A, PROMPT_B, PROMPT_C

PROMPTS = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
}

LABELS = ["positive", "negative", "neutral"]
DATASET_PATH = Path("data/test_comments.csv")


def evaluate(prompt_name: str, model: str, limit: int | None = None):
    """Evaluate one prompt on the labeled customer comments."""

    df = pd.read_csv(DATASET_PATH)

    if limit is not None:
        df = df.head(limit)

    safe_model = model.replace("/", "_")
    results_path = Path(f"results_{prompt_name}_{safe_model}.csv")

    # Read previously saved predictions to avoid repeated API calls.
    if results_path.exists():
        results = pd.read_csv(results_path)
        print(f"Loaded existing results from {results_path}")
    else:
        results = pd.DataFrame(
            columns=["id", "comment", "label", "prediction"]
        )

    completed_ids = set(results["id"].tolist())
    start_time = time.perf_counter()

    for _, row in df.iterrows():
        comment_id = row["id"]

        if comment_id in completed_ids:
            continue

        answer = None

        # Retry temporary API failures up to 3 times.
        for attempt in range(1, 4):
            try:
                answer = ask_llm(
                    system_prompt=PROMPTS[prompt_name],
                    user_message=row["comment"],
                    model=model,
                    temperature=0,
                )
                break
            except Exception as error:
                print(f"API error on comment {comment_id}: {error}")

                if attempt == 3:
                    raise

                time.sleep(2)

        prediction = parse_label(answer)

        new_result = pd.DataFrame([{
            "id": comment_id,
            "comment": row["comment"],
            "label": row["label"],
            "prediction": prediction,
        }])

        results = pd.concat(
            [results, new_result],
            ignore_index=True,
        )

        # Save after every successful API call.
        results.to_csv(results_path, index=False)

        print(
            f"Comment {comment_id}: "
            f"expected={row['label']}, predicted={prediction}"
        )

        time.sleep(0.5)

    elapsed = time.perf_counter() - start_time

    # Evaluate only the comments included in this run.
    evaluated = df[["id"]].merge(results, on="id", how="inner")

    accuracy = accuracy_score(
        evaluated["label"],
        evaluated["prediction"],
    )

    majority_label = df["label"].value_counts().idxmax()
    baseline = (df["label"] == majority_label).mean()

    invalid_count = (evaluated["prediction"] == "invalid").sum()

    matrix = confusion_matrix(
        evaluated["label"],
        evaluated["prediction"],
        labels=LABELS + ["invalid"],
    )

    print("\n===== Evaluation Results =====")
    print(f"Prompt: {prompt_name}")
    print(f"Model: {model}")
    print(f"Evaluated comments: {len(evaluated)}/{len(df)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Baseline accuracy: {baseline:.2%}")
    print(f"Invalid predictions: {invalid_count}")
    print(f"Time for new API calls: {elapsed:.2f} seconds")

    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=LABELS + ["invalid"],
            columns=LABELS + ["invalid"],
        )
    )

    wrong = evaluated[
        evaluated["label"] != evaluated["prediction"]
    ]

    print("\nWrong predictions:")
    print(wrong.to_string(index=False))

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        choices=["A", "B", "C"],
        required=True,
    )

    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    evaluate(
        prompt_name=args.prompt,
        model=args.model,
        limit=args.limit,
    )
