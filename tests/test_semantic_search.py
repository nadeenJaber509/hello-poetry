
import numpy as np
import pandas as pd

from compare import keyword_search
from semantic_search import semantic_search


class FakeModel:
    """A small fake model for testing without loading a real model."""

    def encode(self, texts, normalize_embeddings=True):
        vectors = {
            "good product": [1.0, 0.0],
            "bad product": [0.0, 1.0],
            "test query": [1.0, 0.0],
        }
        return np.array([vectors[text] for text in texts])


def test_search_with_fake_model():
    df = pd.DataFrame({
        "id": [1, 2],
        "comment": ["good product", "bad product"],
        "label": ["positive", "negative"],
    })
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    results = semantic_search(
        "test query",
        df,
        embeddings,
        FakeModel(),
        top_k=1,
        min_score=0.0,
    )

    assert len(results) == 1
    assert results.iloc[0]["comment"] == "good product"


def test_search_k_larger_than_comments():
    df = pd.DataFrame({
        "id": [1, 2],
        "comment": ["good product", "bad product"],
        "label": ["positive", "negative"],
    })
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    results = semantic_search(
        "test query",
        df,
        embeddings,
        FakeModel(),
        top_k=10,
        min_score=0.0,
    )

    assert len(results) == 2


def test_search_empty_query():
    df = pd.DataFrame({
        "id": [1],
        "comment": ["good product"],
        "label": ["positive"],
    })
    embeddings = np.array([[1.0, 0.0]])

    results = semantic_search(
        "",
        df,
        embeddings,
        FakeModel(),
    )

    assert results.empty

def test_keyword_search_ignores_capitalization():

    comments = [
        "I received the INVOICE by email.",
        "The delivery was late.",
        "The invoice was incorrect.",
    ]

    results = keyword_search(comments, "invoice", top_k=3)

    assert len(results) == 2
    assert all(
        "invoice" in comments[index].lower()
        for index, score in results
    )
