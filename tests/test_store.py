
import numpy as np

import store


class FakeCollection:
    def __init__(self, documents=None, metadatas=None):
        self.documents = documents or []
        self.metadatas = metadatas or []

    def count(self):
        return len(self.documents)

    def get(self, include=None):
        return {
            "documents": self.documents,
            "metadatas": self.metadatas,
        }


class FakeModel:
    def encode(self, texts):
        return np.array([
            [1.0, 0.0] if "poetry" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ])


def make_collection(documents):
    metadatas = [
        {"source": f"doc{i}.md", "title": f"Document {i}"}
        for i in range(len(documents))
    ]
    return FakeCollection(documents, metadatas)


def test_keyword_search_prioritizes_exact_cli_option():
    collection = make_collection([
        "Run poetry install --no-root to install dependencies.",
        "Run poetry install to install dependencies.",
    ])

    results = store.keyword_search(
        "poetry install --no-root",
        collection,
        top_k=2,
    )

    assert len(results) == 2
    assert "--no-root" in results[0]["text"]
    assert results[0]["keyword_score"] > results[1]["keyword_score"]


def test_keyword_search_returns_empty_for_empty_query():
    collection = make_collection([
        "Run poetry install to install dependencies."
    ])

    assert store.keyword_search("the and is", collection) == []


def test_hybrid_search_returns_empty_for_empty_collection():
    collection = make_collection([])

    results = store.hybrid_search(
        "poetry install",
        collection,
        FakeModel(),
    )

    assert results == []


def test_hybrid_search_merges_duplicate_results(monkeypatch):
    text = "Run poetry install to install dependencies."
    collection = make_collection([text])

    semantic_result = {
        "text": text,
        "source": "doc0.md",
        "title": "Document 0",
        "similarity": 0.95,
    }

    monkeypatch.setattr(
        store,
        "search_documents",
        lambda **kwargs: [semantic_result],
    )

    results = store.hybrid_search(
        "poetry install",
        collection,
        FakeModel(),
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["text"] == text
    assert results[0]["rrf_score"] > 0


def test_hybrid_search_rejects_invalid_similarity():
    collection = make_collection([
        "Run poetry install to install dependencies."
    ])

    try:
        store.hybrid_search(
            "poetry install",
            collection,
            FakeModel(),
            min_similarity=1.5,
        )
    except ValueError as exc:
        assert "min_similarity" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
