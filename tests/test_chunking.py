
"""Unit tests for document chunking."""

import pytest

from chunking import chunk_documents, chunk_text


def test_chunk_text_with_overlap():
    chunks = chunk_text("ABCDEFGHIJKLMNO", 6, 2)

    assert chunks == ["ABCDEF", "EFGHIJ", "IJKLMN", "MNO"]


def test_chunk_text_without_overlap():
    text = "ABCDEFGHIJK"
    chunks = chunk_text(text, 5, 0)

    assert chunks == ["ABCDE", "FGHIJ", "K"]
    assert "".join(chunks) == text


def test_empty_text():
    assert chunk_text("", 300, 60) == []


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (300, 300),
        (300, 400),
        (300, -1),
        (0, 0),
    ],
)
def test_invalid_chunk_settings(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("Hello world", chunk_size, overlap)


def test_chunk_size_limit():
    chunks = chunk_text("A" * 1000, 300, 60)

    assert all(len(chunk) <= 300 for chunk in chunks)
    assert chunks[-1].endswith("A")


def test_chunk_documents_metadata_and_unique_ids():
    documents = [
        {
            "source": "example.md",
            "title": "Example",
            "text": "ABCDEFGHIJ",
        },
        {
            "source": "guide.md",
            "title": "Guide",
            "text": "KLMNOPQRST",
        },
    ]

    chunks = chunk_documents(documents, 6, 2)
    ids = [chunk["id"] for chunk in chunks]

    assert len(ids) == len(set(ids))
    assert all(
        set(chunk) == {
            "id",
            "text",
            "source",
            "title",
            "chunk_index",
        }
        for chunk in chunks
    )

    assert chunks[0] == {
        "id": "example.md::0",
        "text": "ABCDEF",
        "source": "example.md",
        "title": "Example",
        "chunk_index": 0,
    }
