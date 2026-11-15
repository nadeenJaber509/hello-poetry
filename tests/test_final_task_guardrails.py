
"""Offline tests for the scope guardrail."""

from unittest.mock import MagicMock, patch

import pytest

from hello_poetry.guardrails import ScopeDecision, check_scope


def test_scope_accepts_poetry_question():
    """A valid JSON response should pass."""
    response = (
        '{"in_scope": true, '
        '"reason": "Poetry dependency management."}'
    )

    with patch(
        "hello_poetry.guardrails.chat_json",
        return_value=mock_chat_completion(response),
    ):
        decision = check_scope(
            "How do I add a dependency using Poetry?",
            model="test-model",
        )

    assert isinstance(decision, ScopeDecision)
    assert decision.in_scope is True


def test_scope_rejects_out_of_scope_question():
    """An out-of-scope classification should be returned."""
    response = (
        '{"in_scope": false, '
        '"reason": "Weather is unrelated to Poetry."}'
    )

    with patch(
        "hello_poetry.guardrails.chat_json",
        return_value=mock_chat_completion(response),
    ):
        decision = check_scope(
            "What is the weather in Ramallah?",
            model="test-model",
        )

    assert decision.in_scope is False


def test_scope_rejects_invalid_boolean():
    """String values must not replace booleans."""
    response = (
        '{"in_scope": "yes", "reason": "Not valid."}'
    )

    with (
        patch(
            "hello_poetry.guardrails.chat_json",
            return_value=mock_chat_completion(response),
        ),
        pytest.raises(ValueError, match="invalid JSON"),
    ):
        check_scope("How do I use Poetry?", model="test-model")


def test_scope_rejects_empty_question():
    """An empty question should raise a clear error."""
    with pytest.raises(
        ValueError,
        match="Question must not be empty",
    ):
        check_scope("   ", model="test-model")

def mock_chat_completion(content):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


from hello_poetry.guardrails import check_evidence


def test_evidence_above_threshold():
    chunks = [
        {"source": "cli.md", "similarity": 0.72},
        {"source": "faq.md", "similarity": 0.51},
    ]

    decision = check_evidence(chunks, threshold=0.46)

    assert decision.has_evidence is True
    assert decision.best_similarity == 0.72


def test_evidence_below_threshold():
    chunks = [
        {"source": "cli.md", "similarity": 0.31},
    ]

    decision = check_evidence(chunks, threshold=0.46)

    assert decision.has_evidence is False


def test_evidence_empty_results():
    decision = check_evidence([], threshold=0.46)

    assert decision.has_evidence is False
    assert decision.best_similarity == 0.0


def test_evidence_at_threshold():
    chunks = [
        {"source": "cli.md", "similarity": 0.46},
    ]

    decision = check_evidence(chunks, threshold=0.46)

    assert decision.has_evidence is True


def test_evidence_invalid_threshold():
    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1",
    ):
        check_evidence([], threshold=1.5)
