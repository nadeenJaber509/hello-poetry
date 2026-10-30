
import csv
import statistics
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_documents
from documents import load_documents
from store import DB_PATH, MODEL_NAME, hybrid_search


DATA_PATH = Path("data")
DOCS_PATH = DATA_PATH / "poetry_docs"
QUESTIONS_PATH = DATA_PATH / "questions.csv"

SETTINGS = [
    (300, 0),
    (300, 60),
    (800, 0),
    (800, 150),
    (1500, 0),
    (1500, 300),
]


def load_questions():
    with QUESTIONS_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def build_collection(name, chunks, model, client):
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts).tolist()

        ids = [
            f"{chunk['source']}::chunk_{i}"
            for i, chunk in enumerate(batch, start=start)
        ]

        metadatas = [
            {
                "source": chunk["source"],
                "title": chunk["title"],
            }
            for chunk in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection


def normalize_text(text):
    text = text.casefold()

    # Remove punctuation and Markdown symbols
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def evaluate_question(question, collection, model):
    query = question["question"]
    expected_source = question.get("expected_source", "").strip()

    key_phrase = question.get(
        "key_phrase",
        question.get("expected_phrase", ""),
    ).strip()

    if not expected_source:
        return None

    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    ranked_results = []

    for i, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        ranked_results.append({
            "source": metadata["source"],
            "title": metadata["title"],
            "text": text,
            "similarity": 1 - distance,
        })

    for rank, result in enumerate(ranked_results, start=1):
        source_matches = result["source"] == expected_source

        normalized_phrase = normalize_text(key_phrase)
        normalized_result = normalize_text(result["text"])

        phrase_parts = normalized_phrase.split()

        phrase_matches = all(
            part in normalized_result.split()
            for part in phrase_parts
        )

        if source_matches and phrase_matches:
            return {
                "hit_rank": rank,
                "results": ranked_results,
            }

    return {
        "hit_rank": None,
        "results": ranked_results,
    }


def evaluate_hybrid_question(question, collection, model):
    query = question["question"]
    expected_source = question.get("expected_source", "").strip()

    key_phrase = question.get(
        "key_phrase",
        question.get("expected_phrase", ""),
    ).strip()

    if not expected_source:
        return None

    ranked_results = hybrid_search(
        query=query,
        collection=collection,
        model=model,
        top_k=5,
    )

    for rank, result in enumerate(ranked_results, start=1):
        source_matches = result["source"] == expected_source

        normalized_phrase = normalize_text(key_phrase)
        normalized_result = normalize_text(result["text"])

        phrase_parts = normalized_phrase.split()

        phrase_matches = all(
            part in normalized_result.split()
            for part in phrase_parts
        )

        if source_matches and phrase_matches:
            return {
                "hit_rank": rank,
                "results": ranked_results,
            }

    return {
        "hit_rank": None,
        "results": ranked_results,
    }


def save_failures(failures, failure_path):
    with failure_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "expected_source",
                "key_phrase",
                "top_3_results",
            ],
        )

        writer.writeheader()

        for failure in failures:
            top_results = " | ".join(
                f"{r['source']}: {r['text'][:200]}"
                for r in failure["top_3_results"]
            )

            writer.writerow({
                "question": failure["question"],
                "expected_source": failure["expected_source"],
                "key_phrase": failure["key_phrase"],
                "top_3_results": top_results,
            })


def main():
    documents = load_documents(DOCS_PATH)
    questions = load_questions()

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)

    summary_rows = []

    eligible_questions = [
        q for q in questions
        if q.get("expected_source", "").strip()
    ]

    for chunk_size, overlap in SETTINGS:
        setting_name = f"poetry_{chunk_size}_{overlap}"

        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        collection = build_collection(
            setting_name,
            chunks,
            model,
            client,
        )

        # Semantic Search counters
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        failures = []

        # Hybrid Search counters
        hybrid_hits_at_1 = 0
        hybrid_hits_at_3 = 0
        hybrid_hits_at_5 = 0
        hybrid_failures = []

        for question in eligible_questions:

            # Semantic Search evaluation
            evaluation = evaluate_question(
                question,
                collection,
                model,
            )

            rank = evaluation["hit_rank"]

            if rank is not None:
                if rank <= 1:
                    hits_at_1 += 1

                if rank <= 3:
                    hits_at_3 += 1

                if rank <= 5:
                    hits_at_5 += 1

            else:
                failures.append({
                    "question": question["question"],
                    "expected_source": question["expected_source"],
                    "key_phrase": question.get(
                        "key_phrase",
                        question.get("expected_phrase", ""),
                    ),
                    "top_3_results": evaluation["results"][:3],
                })

            # Hybrid Search evaluation
            hybrid_evaluation = evaluate_hybrid_question(
                question,
                collection,
                model,
            )

            hybrid_rank = hybrid_evaluation["hit_rank"]

            if hybrid_rank is not None:
                if hybrid_rank <= 1:
                    hybrid_hits_at_1 += 1

                if hybrid_rank <= 3:
                    hybrid_hits_at_3 += 1

                if hybrid_rank <= 5:
                    hybrid_hits_at_5 += 1

            else:
                hybrid_failures.append({
                    "question": question["question"],
                    "expected_source": question["expected_source"],
                    "key_phrase": question.get(
                        "key_phrase",
                        question.get("expected_phrase", ""),
                    ),
                    "top_3_results": hybrid_evaluation["results"][:3],
                })

        total = len(eligible_questions)

        # Calculate Semantic Search metrics
        semantic_row = {
            "setting": setting_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "number_of_chunks": len(chunks),
            "average_chunk_length": round(
                statistics.mean(
                    len(c["text"]) for c in chunks
                ), 2
            ) if chunks else 0,
            "hit_at_1": round(
                hits_at_1 / total, 4
            ) if total else 0,
            "hit_at_3": round(
                hits_at_3 / total, 4
            ) if total else 0,
            "hit_at_5": round(
                hits_at_5 / total, 4
            ) if total else 0,
        }

        # Calculate Hybrid Search metrics
        hybrid_row = {
            "setting": setting_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "number_of_chunks": len(chunks),
            "average_chunk_length": semantic_row[
                "average_chunk_length"
            ],
            "hit_at_1": round(
                hybrid_hits_at_1 / total, 4
            ) if total else 0,
            "hit_at_3": round(
                hybrid_hits_at_3 / total, 4
            ) if total else 0,
            "hit_at_5": round(
                hybrid_hits_at_5 / total, 4
            ) if total else 0,
        }

        summary_rows.append({
            "search_type": "Semantic",
            **semantic_row,
        })

        summary_rows.append({
            "search_type": "Hybrid",
            **hybrid_row,
        })

        # Save Semantic failures
        failure_path = DATA_PATH / f"failures_{setting_name}.csv"
        save_failures(failures, failure_path)

        # Save Hybrid failures
        hybrid_failure_path = (
            DATA_PATH / f"hybrid_failures_{setting_name}.csv"
        )
        save_failures(hybrid_failures, hybrid_failure_path)

        print(f"\nSetting: {setting_name}")
        print("Chunks:", len(chunks))
        print(
            "Average chunk length:",
            semantic_row["average_chunk_length"],
        )

        print("\nSemantic Search:")
        print("Hit@1:", semantic_row["hit_at_1"])
        print("Hit@3:", semantic_row["hit_at_3"])
        print("Hit@5:", semantic_row["hit_at_5"])
        print("Failures saved:", failure_path)

        print("\nHybrid Search:")
        print("Hit@1:", hybrid_row["hit_at_1"])
        print("Hit@3:", hybrid_row["hit_at_3"])
        print("Hit@5:", hybrid_row["hit_at_5"])
        print("Failures saved:", hybrid_failure_path)

    # Save all experiment results
    summary_path = DATA_PATH / "experiment_results.csv"

    with summary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nAll experiments completed.")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
