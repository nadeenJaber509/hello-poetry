
import numpy as np
import pytest

from search import keyword_search, search
from similarity import cosine_similarity


class FakeModel:
    """Return fixed query embeddings without loading a real model."""

    def encode(self, sentences):
        return np.array([[1.0, 0.0] for _ in sentences])


def test_cosine_identical_vectors():
    result = cosine_similarity([1, 0], [1, 0])
    assert result == pytest.approx(1.0)


def test_cosine_perpendicular_vectors():
    result = cosine_similarity([1, 0], [0, 1])
    assert result == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    result = cosine_similarity([1, 0], [-1, 0])
    assert result == pytest.approx(-1.0)


def test_semantic_search_ranking():
    comments = ["Best match", "No match", "Opposite"]
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    results = search(
        "test query",
        comments,
        embeddings,
        FakeModel(),
        k=3,
    )

    assert [item["index"] for item in results] == [0, 1, 2]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)
    assert results[2]["score"] == pytest.approx(-1.0)


def test_k_larger_than_comment_count():
    comments = ["First", "Second"]
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    results = search(
        "test",
        comments,
        embeddings,
        FakeModel(),
        k=10,
    )

    assert len(results) == 2


def test_keyword_search_case_insensitive():
    comments = ["FAST DELIVERY", "slow shipping"]

    results = keyword_search("fast delivery", comments, k=2)

    assert results[0]["index"] == 0
    assert results[0]["score"] == 2


def test_empty_query():
    comments = ["Some comment"]
    embeddings = np.array([[1.0, 0.0]])

    assert search("", comments, embeddings, FakeModel()) == []
    assert keyword_search("", comments) == []
