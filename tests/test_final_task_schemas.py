
"""Offline tests for structured Poetry support answers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hello_poetry.schemas import get_structured_answer


def fake_response(content):
    """Simulate the structure of a Groq response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


VALID_JSON = """
{
    "answerable": true,
    "answer": "Use poetry add.",
    "commands": ["poetry add requests"],
    "sources": ["cli.md"],
    "confidence": "high"
}
"""

INVALID_JSON = """
{
    "answerable": true,
    "answer": "Use poetry add.",
    "commands": [],
    "sources": [],
    "confidence": "very high"
}
"""


def test_valid_json_first_try(tmp_path, monkeypatch):
    """Valid response should pass without retries."""
    monkeypatch.setattr(
        "hello_poetry.schemas.RESULTS_DIR",
        tmp_path,
    )

    with patch(
        "hello_poetry.schemas.chat_json",
        return_value=fake_response(VALID_JSON),
    ) as mock_chat:
        answer, retries = get_structured_answer(
            [{"role": "user", "content": "How to add a dependency?"}],
            model="fake-model",
        )

    assert answer.answerable is True
    assert answer.confidence == "high"
    assert retries == 0
    assert mock_chat.call_count == 1


def test_invalid_then_valid(tmp_path, monkeypatch):
    """Invalid response should trigger a retry."""
    monkeypatch.setattr(
        "hello_poetry.schemas.RESULTS_DIR",
        tmp_path,
    )

    with patch(
        "hello_poetry.schemas.chat_json",
        side_effect=[
            fake_response(INVALID_JSON),
            fake_response(VALID_JSON),
        ],
    ) as mock_chat:
        answer, retries = get_structured_answer(
            [{"role": "user", "content": "How to add a dependency?"}],
            model="fake-model",
            max_retries=2,
        )

    assert answer.answer == "Use poetry add."
    assert retries == 1
    assert mock_chat.call_count == 2


def test_invalid_after_max_retries(tmp_path, monkeypatch):
    """Repeated invalid responses must raise an error."""
    monkeypatch.setattr(
        "hello_poetry.schemas.RESULTS_DIR",
        tmp_path,
    )

    with (
        patch(
            "hello_poetry.schemas.chat_json",
            return_value=fake_response(INVALID_JSON),
        ) as mock_chat,
        pytest.raises(
            ValueError,
            match="Invalid structured answer",
        ),
    ):
        get_structured_answer(
            [{"role": "user", "content": "Test"}],
            model="fake-model",
            max_retries=2,
        )

    assert mock_chat.call_count == 3
