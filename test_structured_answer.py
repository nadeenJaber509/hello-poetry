import pytest
from pydantic import ValidationError

from schemas import SupportAnswer


def test_valid_support_answer():
    data = {
        "answerable": True,
        "answer": "Use poetry add requests.",
        "commands": ["poetry add requests"],
        "sources": ["basic-usage.md"],
        "confidence": "high",
    }

    result = SupportAnswer.model_validate(data)

    assert result.answerable is True
    assert result.confidence == "high"
    assert result.commands == ["poetry add requests"]


def test_invalid_confidence():
    data = {
        "answerable": True,
        "answer": "Install a package.",
        "commands": [],
        "sources": ["basic-usage.md"],
        "confidence": "very high",
    }

    with pytest.raises(ValidationError):
        SupportAnswer.model_validate(data)
def test_missing_required_field():
    data = {
        "answerable": True,
        "answer": "Install a package.",
        "commands": [],
        "confidence": "high",
    }

    with pytest.raises(ValidationError):
        SupportAnswer.model_validate(data)