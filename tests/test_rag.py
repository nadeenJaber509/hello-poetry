
from unittest.mock import Mock, patch

import pytest

from hello_poetry.rag import (
    answer_question,
    build_messages,
    format_sources,
)


@pytest.fixture
def sample_sources():
    return [
        {
            "title": "Commands",
            "source": "cli.md",
            "text": "Use poetry add requests to add a dependency.",
            "similarity": 0.9,
        },
        {
            "title": "Managing Dependencies",
            "source": "managing-dependencies.md",
            "text": "Dependencies are stored in pyproject.toml.",
            "similarity": 0.8,
        },
    ]


def test_format_sources(sample_sources):
    result = format_sources(sample_sources)

    assert "[1]" in result
    assert "[2]" in result
    assert "Commands" in result
    assert "cli.md" in result
    assert "poetry add requests" in result


def test_format_sources_empty():
    result = format_sources([])

    assert result == ""


def test_build_messages(sample_sources):
    system_prompt, user_message = build_messages(
        question="How do I add a dependency?",
        results=sample_sources,
    )

    assert "Poetry" in system_prompt
    assert "numbered sources" in system_prompt
    assert "[1]" in user_message
    assert "[2]" in user_message
    assert "How do I add a dependency?" in user_message
    assert "poetry add requests" in user_message


@patch("hello_poetry.rag.ask_llm")
@patch("hello_poetry.rag.query_collection")
@patch("hello_poetry.rag.get_embedding_model")
def test_answer_question(
    mock_embedding,
    mock_query,
    mock_llm,
    sample_sources,
):
    mock_embedding.return_value = Mock()
    mock_query.return_value = sample_sources
    mock_llm.return_value = "Run poetry add requests. [1]"

    result = answer_question(
        question="How do I add a dependency?",
        collection_name="poetry_800_150",
        k=3,
    )

    assert result["answer"] == "Run poetry add requests. [1]"
    assert result["sources"] == sample_sources
    assert "prompt" in result

    mock_embedding.assert_called_once()

    mock_query.assert_called_once_with(
        name="poetry_800_150",
        query="How do I add a dependency?",
        model=mock_embedding.return_value,
        k=3,
    )

    mock_llm.assert_called_once()

    assert mock_llm.call_args.kwargs["temperature"] == 0
    assert (
        mock_llm.call_args.kwargs["model"]
        == "openai/gpt-oss-20b"
    )


@pytest.mark.parametrize(
    "question",
    ["", " ", "\n"],
)
def test_empty_question(question):
    with pytest.raises(ValueError):
        answer_question(
            question=question,
            collection_name="poetry_800_150",
        )


@pytest.mark.parametrize(
    "k",
    [0, -1],
)
def test_invalid_k(k):
    with pytest.raises(ValueError):
        answer_question(
            question="How do I add a dependency?",
            collection_name="poetry_800_150",
            k=k,
        )


@patch("hello_poetry.rag.ask_llm")
@patch("hello_poetry.rag.query_collection")
@patch("hello_poetry.rag.get_embedding_model")
def test_no_sources(
    mock_embedding,
    mock_query,
    mock_llm,
):
    mock_query.return_value = []

    result = answer_question(
        question="Unknown question",
        collection_name="poetry_800_150",
        k=3,
    )

    assert result["sources"] == []
    assert result["answer"] == "No relevant documentation was found."
    assert "prompt" in result

    mock_embedding.assert_called_once()
    mock_query.assert_called_once()
    mock_llm.assert_not_called()


@patch("hello_poetry.rag.ask_llm")
@patch("hello_poetry.rag.query_collection")
@patch("hello_poetry.rag.get_embedding_model")
def test_custom_k(
    mock_embedding,
    mock_query,
    mock_llm,
    sample_sources,
):
    mock_query.return_value = sample_sources
    mock_llm.return_value = "Test answer. [1]"

    result = answer_question(
        question="Test question",
        collection_name="poetry_800_150",
        k=5,
    )

    assert result["answer"] == "Test answer. [1]"
    assert mock_query.call_args.kwargs["k"] == 5


@patch("hello_poetry.rag.ask_llm")
@patch("hello_poetry.rag.query_collection")
@patch("hello_poetry.rag.get_embedding_model")
def test_llm_uses_retrieved_sources(
    mock_embedding,
    mock_query,
    mock_llm,
    sample_sources,
):
    mock_query.return_value = sample_sources
    mock_llm.return_value = "Use poetry add. [1]"

    answer_question(
        question="How do I add a dependency?",
        collection_name="poetry_800_150",
    )

    user_message = mock_llm.call_args.kwargs["user_message"]

    assert "cli.md" in user_message
    assert "managing-dependencies.md" in user_message
    assert "poetry add requests" in user_message
    assert "How do I add a dependency?" in user_message
