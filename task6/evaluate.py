
import re
import time
from pathlib import Path

import pandas as pd
from llm import ask_llm, parse_label
from prompts import PROMPT_A, PROMPT_B, PROMPT_C
from sklearn.metrics import confusion_matrix

DATA_PATH = Path(__file__).parent / "data" / "test_comments.csv"
RESULTS_DIR = Path(__file__).parent

MODEL = "openai/gpt-oss-20b"
LABELS = ["positive", "negative", "neutral"]
MATRIX_LABELS = LABELS + ["invalid"]

PROMPTS = {
    "A_zero_shot": PROMPT_A,
    "B_few_shot": PROMPT_B,
    "C_improved": PROMPT_C,
}


def majority_baseline(labels):
    """Return the accuracy of always predicting the most common label."""
    return labels.value_counts(normalize=True).max()


def evaluate_prompt(data, prompt_name, system_prompt):
    """Evaluate one system prompt on the full dataset."""
    predictions = []
    start_time = time.perf_counter()

    for _, row in data.iterrows():
        prediction = "invalid"

        # Retry the API request up to three times.
        for attempt in range(3):
            try:
                response = ask_llm(
                    system_prompt=system_prompt,
                    user_message=row["comment"],
                    model=MODEL,
                    temperature=0,
                )
                prediction = parse_label(response)
                break
            except (ValueError, RuntimeError, OSError) as error:
                print(
                    f"Attempt {attempt + 1}/3 failed "
                    f"for comment {row['id']}: {error}"
                )
                if attempt < 2:
                    time.sleep(0.5)

        predictions.append(prediction)

        # Pause between comments to reduce request frequency.
        time.sleep(0.5)

    runtime = time.perf_counter() - start_time

    results = data.copy()
    results["prediction"] = predictions
    results["correct"] = results["label"] == results["prediction"]

    # Invalid predictions count as wrong.
    accuracy = results["correct"].mean()
    invalid_count = (results["prediction"] == "invalid").sum()

    print(f"\n--- {prompt_name} | {MODEL} ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Invalid predictions: {invalid_count}/{len(data)}")
    print(f"Runtime: {runtime:.2f} seconds")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(
        confusion_matrix(
            results["label"],
            results["prediction"],
            labels=MATRIX_LABELS,
        )
    )

    safe_model_name = re.sub(r"[^A-Za-z0-9_-]", "_", MODEL)
    results_path = (
        RESULTS_DIR
        / f"results_{prompt_name}_{safe_model_name}.csv"
    )
    results.to_csv(results_path, index=False)

    print(f"Saved results to: {results_path}")

    return accuracy, invalid_count, runtime


def main():
    data = pd.read_csv(DATA_PATH)

    required_columns = {"id", "comment", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"CSV must contain these columns: {sorted(required_columns)}"
        )

    if data["comment"].isna().any() or data["label"].isna().any():
        raise ValueError("The CSV contains missing comments or labels.")

    unexpected_labels = set(data["label"]) - set(LABELS)
    if unexpected_labels:
        raise ValueError(f"Unexpected labels found: {unexpected_labels}")

    print(f"Loaded {len(data)} comments.")
    print(f"Model: {MODEL}")
    print(f"Majority-class baseline: {majority_baseline(data['label']):.2%}")

    summary = []

    for name, system_prompt in PROMPTS.items():
        accuracy, invalid_count, runtime = evaluate_prompt(
            data, name, system_prompt
        )
        summary.append(
            {
                "prompt": name,
                "model": MODEL,
                "accuracy": accuracy,
                "invalid_predictions": invalid_count,
                "runtime_seconds": round(runtime, 2),
            }
        )

    print("\n--- Evaluation Summary ---")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    evaluate_prompt(data, "B_few_shot", PROMPT_B)