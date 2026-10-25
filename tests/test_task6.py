
from unittest.mock import MagicMock, patch

import pytest

from task6.llm import ask_llm, parse_label


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"label": "positive"}', "positive"),
        ('{"label": " Positive "}', "positive"),
        ('{"label": "NEGATIVE"}', "negative"),
        ('{"label": "neutral"}', "neutral"),
        ("Sure! Here is the label: positive", "invalid"),
        ('{"label": "happy"}', "invalid"),
        ("", "invalid"),
        ("not json", "invalid"),
        ('{"other": "positive"}', "invalid"),
    ],
)
def test_parse_label(response, expected):
    assert parse_label(response) == expected


def test_ask_llm_with_mock_client():
    fake_response = MagicMock()
    fake_response.choices[0].message.content = '{"label": "positive"}'

    with (
        patch("task6.llm.Groq") as mock_groq,
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
    ):
        mock_client = mock_groq.return_value
        mock_client.chat.completions.create.return_value = fake_response

        result = ask_llm(
            system_prompt="Classify sentiment.",
            user_message="I love this app!",
            model="test-model",
            temperature=0,
        )

        assert result == '{"label": "positive"}'
        mock_groq.assert_called_once_with(api_key="test-key")
        mock_client.chat.completions.create.assert_called_once()
