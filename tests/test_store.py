"""Tests for persistent document storage and retrieval."""

import numpy as np

from hello_poetry.store import build_collection, query_collection


class FakeModel:
    """Return simple deterministic embeddings for testing."""

    def encode(self, texts, **kwargs):
        vectors = []

        for text in texts:
            if "dependency" in text.lower():
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])

        return np.array(vectors, dtype=float)


def test_build_and_query_collection(tmp_path):
    model = FakeModel()

    chunks = [
        {
            "id": "cli.md::0",
            "text": "Add a dependency using poetry add.",
            "source": "cli.md",
            "title": "Commands",
            "chunk_index": 0,
        },
        {
            "id": "faq.md::0",
            "text": "Poetry supports virtual environments.",
            "source": "faq.md",
            "title": "FAQ",
            "chunk_index": 0,
        },
    ]

    collection = build_collection(
        "test_docs",
        chunks,
        model,
        path=tmp_path,
    )

    assert collection.count() == 2

    results = query_collection(
        "test_docs",
        "How do I add a dependency?",
        model,
        k=1,
        path=tmp_path,
    )

    assert len(results) == 1
    assert results[0]["source"] == "cli.md"
    assert results[0]["chunk_index"] == 0
    assert results[0]["similarity"] > 0.9

    # Rebuilding should not create duplicate records.
    rebuilt = build_collection(
        "test_docs",
        chunks,
        model,
        path=tmp_path,
    )
    assert rebuilt.count() == 2


def test_empty_query_returns_no_results(tmp_path):
    results = query_collection(
        "unused_collection",
        "   ",
        FakeModel(),
        path=tmp_path,
    )

    assert results == []
