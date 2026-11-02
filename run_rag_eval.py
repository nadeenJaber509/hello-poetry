
import argparse
import csv
import time
from pathlib import Path

from groq import APIError

from rag import answer_question
from store import build_vector_store
from task6.llm import ask_llm

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "data" / "questions.csv"

PLAIN_SYSTEM_PROMPT = (
    "You answer questions about Poetry, the Python dependency manager. "
    "Answer clearly and briefly. Include the command when there is one. "
    "If you do not know the answer, say so."
)


def load_questions():
    """Read evaluation questions from the CSV file."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def call_with_retries(function, *args, max_retries=3, **kwargs):
    """Retry an API call up to three times after the first attempt."""
    for attempt in range(max_retries + 1):
        try:
            return function(*args, **kwargs)
        except Exception:
            if attempt == max_retries:
                raise

            print(f"Request failed. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(0.5)


import re


def check_retrieval_hit(question, sources):
    """Check whether the expected source contains all required key phrases."""

    if question["answerable"].strip().lower() != "yes":
        return "n/a"

    expected_source = question["expected_source"].strip().lower()
    key_phrase = question["key_phrase"].strip().lower()

    matching_text = " ".join(
        source["text"].lower()
        for source in sources
        if Path(source["source"]).name.lower() == expected_source
    )

    if not matching_text:
        return "no"

    # Support multiple phrases separated by commas or semicolons.
    phrases = [
        phrase.strip()
        for phrase in re.split(r"[;,]", key_phrase)
        if phrase.strip()
    ]

    return "yes" if all(
        phrase in matching_text for phrase in phrases
    ) else "no"


def evaluate_question(question, mode, collection=None, embedding_model=None, k=3):
    """Generate an answer in plain or RAG mode."""

    if mode == "plain":
        answer = call_with_retries(
            ask_llm,
            system_prompt=PLAIN_SYSTEM_PROMPT,
            user_message=question["question"],
            model=MODEL,
            temperature=TEMPERATURE,
        )
        sources = []

    else:
        result = call_with_retries(
            answer_question,
            question=question["question"],
            collection=collection,
            model=embedding_model,
            k=k,
        )
        answer = result["answer"]
        sources = result["sources"]

    retrieved_sources = "; ".join(
        dict.fromkeys(source["source"] for source in sources)
    )

    retrieval_hit = (
        check_retrieval_hit(question, sources)
        if mode == "rag"
        else ""
    )

    return {
        "question": question["question"],
        "expected_source": question["expected_source"],
        "key_phrase": question["key_phrase"],
        "difficulty": question["difficulty"],
        "answerable": question["answerable"],
        "answer": answer,
        "retrieved_sources": retrieved_sources,
        "retrieval_hit": retrieval_hit,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Poetry questions using plain LLM or RAG."
    )

    parser.add_argument(
        "--mode",
        choices=["plain", "rag"],
        required=True,
        help="Choose plain or RAG evaluation.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of sources to retrieve (default: 3).",
    )

    args = parser.parse_args()

    if args.k < 0:
        parser.error("--k must be zero or greater.")

    questions = load_questions()

    print(f"Loaded {len(questions)} questions.")
    print(f"Evaluation mode: {args.mode}")
    print(f"Number of retrieved sources (k): {args.k}")

    collection = None
    embedding_model = None

    if args.mode == "rag":
        collection, embedding_model = build_vector_store()

    results = []

    for index, question in enumerate(questions, start=1):
        print(f"\n[{index}/{len(questions)}] {question['question']}")

        try:
            result = evaluate_question(
                question=question,
                mode=args.mode,
                collection=collection,
                embedding_model=embedding_model,
                k=args.k,
            )

            results.append({
                "id": index,
                **result,
            })

            print("Answer:", result["answer"])
            print("Retrieved sources:", result["retrieved_sources"] or "None")
            print("Retrieval hit:", result["retrieval_hit"] or "N/A")

        except (APIError, ValueError) as error:
            print(f"Error evaluating question: {error}")

            results.append({
                "id": index,
                "question": question["question"],
                "expected_source": question["expected_source"],
                "key_phrase": question["key_phrase"],
                "difficulty": question["difficulty"],
                "answerable": question["answerable"],
                "answer": f"ERROR: {error}",
                "retrieved_sources": "",
                "retrieval_hit": "error",
            })

        time.sleep(0.5)

    output_file = BASE_DIR / f"answers_{args.mode}_k{args.k}.csv"

    fieldnames = [
        "id",
        "question",
        "expected_source",
        "key_phrase",
        "difficulty",
        "answerable",
        "answer",
        "retrieved_sources",
        "retrieval_hit",
    ]

    with open(output_file, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
