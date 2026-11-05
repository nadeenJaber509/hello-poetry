"""Evaluate plain Groq answers and Poetry RAG answers on the same questions.

Run from the project root:
    poetry run python -m scripts.task9.run_rag_eval --mode plain --limit 1
    poetry run python -m scripts.task9.run_rag_eval --mode rag --k 3 --limit 1
"""

import argparse
import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from hello_poetry.llm import ask_llm
from hello_poetry.rag import answer_question

MODEL_NAME = "openai/gpt-oss-20b"
COLLECTION_NAME = "poetry_800_150"
TEMPERATURE = 0
PLAIN_SYSTEM_PROMPT = (
    "You answer questions about Poetry, the Python dependency manager. "
    "Keep the answer short, and include the command when there is one."
)

INPUT_COLUMNS = ("id", "question", "expected_source", "key_phrase")
OUTPUT_COLUMNS = (*INPUT_COLUMNS, "answer", "retrieved_sources", "retrieval_hit")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "task9"


def load_questions(path: Path) -> list[dict[str, str]]:
    """Load the evaluation questions and validate required columns and IDs."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(INPUT_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Questions CSV must contain: {', '.join(INPUT_COLUMNS)}")
        questions = list(reader)

    ids = [str(q["id"]).strip() for q in questions]
    if not questions or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Questions CSV must have at least one row and unique nonempty IDs.")
    if any(not str(q["question"]).strip() for q in questions):
        raise ValueError("Each question must contain nonempty text.")
    return questions


def retrieval_hit(results: list[dict], expected_source: str, key_phrase: str) -> str:
    """Return true/false only if one chunk matches source AND phrase."""
    if not expected_source.strip():
        return ""
    if not key_phrase.strip():
        raise ValueError("Answerable questions require a key_phrase.")
    hit = any(
        str(chunk.get("source", "")).strip() == expected_source.strip()
        and key_phrase.casefold() in str(chunk.get("text", "")).casefold()
        for chunk in results
    )
    return "true" if hit else "false"


def load_saved_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read previously saved answers so reruns do not repeat API requests."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(OUTPUT_COLUMNS):
            raise ValueError(f"Existing CSV has an unexpected header: {path}")
        rows = list(reader)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs in saved results.")
    return {row["id"]: row for row in rows}


def save_rows(path: Path, rows: dict[str, dict[str, str]]) -> None:
    """Persist each completed answer using atomic file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())
    temporary_path.replace(path)


def get_answer(question: str, mode: str, k: int) -> tuple[str, list[dict]]:
    """Run exactly one of the two evaluation modes."""
    if mode == "plain":
        return (
            ask_llm(
                system_prompt=PLAIN_SYSTEM_PROMPT,
                user_message=question,
                model=MODEL_NAME,
                temperature=TEMPERATURE,
            ),
            [],
        )
    result = answer_question(question, COLLECTION_NAME, k=k)
    return result["answer"], result["sources"]


def get_answer_with_retry(
    question: str, mode: str, k: int, attempts: int = 3
) -> tuple[str, list[dict]]:
    """Retry transient API failures, but not local configuration mistakes."""
    from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    retryable = (APIConnectionError, APITimeoutError, RateLimitError)
    for attempt in range(1, attempts + 1):
        try:
            answer, sources = get_answer(question, mode, k)
            if answer.strip():
                return answer, sources

            if attempt == attempts:
                raise ValueError(
                    f"Model returned an empty answer after {attempts} attempts."
                )

            print(f"Empty answer (attempt {attempt}/{attempts}). Retrying...")
        except APIStatusError as error:
            if error.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                raise
            if attempt == attempts:
                raise
            print(f"Temporary API error (attempt {attempt}/{attempts}): {error}")
        except retryable as error:
            if attempt == attempts:
                raise
            print(f"Temporary API error (attempt {attempt}/{attempts}): {error}")
        time.sleep(1.0 * attempt)
    raise RuntimeError("Retry attempts exhausted")


def run(mode: str, k: int = 3, limit: int | None = None) -> Path:
    """Run pending questions and save completed answers immediately."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    questions = load_questions(QUESTIONS_PATH)
    path = RESULTS_DIR / f"answers_{mode}_k{k}.csv"
    saved = load_saved_rows(path)
    selected = questions[:limit] if limit is not None else questions

    # Never silently reuse an answer for a question whose wording changed.
    for row in selected:
        previous = saved.get(row["id"])
        if previous and any(previous[c] != str(row[c]) for c in INPUT_COLUMNS):
            raise ValueError(
                f"Question {row['id']} changed since its answer was saved. "
                f"Review {path} before rerunning."
            )

    pending = [row for row in selected if row["id"] not in saved]
    print(f"Mode: {mode}; k={k}; dataset size: {len(questions)}")
    if len(questions) != 30:
        print("WARNING: Task 9 final evaluation requires 30 verified questions.")
    print(f"Already saved: {len(selected) - len(pending)}; pending: {len(pending)}")

    if not pending:
        print(f"Nothing to call. Results: {path}")
        return path

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing from the environment or .env")

    try:
        for index, question_row in enumerate(pending, start=1):
            if index > 1:
                time.sleep(0.5)
            question = str(question_row["question"])
            print(f"[{index}/{len(pending)}] {question_row['id']}: {question}")
            answer, sources = get_answer_with_retry(question, mode, k)
            if not answer.strip():
                raise ValueError(f"Empty answer returned for {question_row['id']}")

            saved[question_row["id"]] = {
                **{column: str(question_row[column]) for column in INPUT_COLUMNS},
                "answer": answer,
                "retrieved_sources": (
                    "; ".join(str(s.get("source", "")) for s in sources)
                    if mode == "rag" else ""
                ),
                "retrieval_hit": (
                    retrieval_hit(
                        sources,
                        str(question_row["expected_source"]),
                        str(question_row["key_phrase"]),
                    )
                    if mode == "rag" else ""
                ),
            }
            save_rows(path, saved)
    except Exception:
        print(f"Stopped; successfully completed rows remain saved in {path}")
        raise

    print(f"Saved {len(saved)} answers to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plain", "rag"), required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N questions (for smoke testing)")
    args = parser.parse_args()
    run(mode=args.mode, k=args.k, limit=args.limit)


if __name__ == "__main__":
    main()
