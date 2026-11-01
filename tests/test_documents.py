
"""Unit tests for loading Markdown documents."""

from documents import load_documents


def test_load_document_with_front_matter(tmp_path):
    document = tmp_path / "example.md"

    document.write_text(
        '---\ntitle: "Example Guide"\ndraft: false\n---\n'
        '\n# Introduction\n\n\n\nHello world.\n',
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "example.md"
    assert documents[0]["title"] == "Example Guide"
    assert documents[0]["text"] == "# Introduction\n\nHello world."


def test_load_document_without_front_matter(tmp_path):
    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nSome content.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents[0]["title"] == "guide.md"
    assert documents[0]["text"] == "# Guide\n\nSome content."


def test_ignore_non_markdown_files(tmp_path):
    (tmp_path / "guide.md").write_text("Hello", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("License text", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "guide.md"


def test_missing_folder(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
