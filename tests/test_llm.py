
from unittest.mock import MagicMock, patch

import pytest

from hello_poetry.llm import ask_llm, parse_label


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"label": "positive"}', "positive"),
        ('{"label": " Negative "}', "negative"),
        ('{"label": "neutral"}', "neutral"),
        ('Sure! Here is the label: positive', "invalid"),
        ('{"label": "unknown"}', "invalid"),
        ("", "invalid"),
        ('{"label": "positive"} Extra text', "invalid"),
    ],
)
def test_parse_label(response, expected):
    assert parse_label(response) == expected


def test_ask_llm_without_internet():
    fake_client = MagicMock()
    fake_response = MagicMock()

    fake_response.choices[0].message.content = (
        '{"label": "positive"}'
    )

    fake_client.chat.completions.create.return_value = (
        fake_response
    )

    with (
        patch(
            "hello_poetry.llm.Groq",
            return_value=fake_client,
        ) as mock_groq,
        patch.dict(
            "os.environ",
            {"GROQ_API_KEY": "fake-test-key"},
        ),
    ):
        answer = ask_llm(
            system_prompt="Classify the comment.",
            user_message="Amazing service!",
            model="openai/gpt-oss-20b",
            temperature=0,
        )

    assert answer == '{"label": "positive"}'
    assert parse_label(answer) == "positive"

    mock_groq.assert_called_once_with(
        api_key="fake-test-key"
    )

    fake_client.chat.completions.create.assert_called_once_with(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "Classify the comment.",
            },
            {
                "role": "user",
                "content": "Amazing service!",
            },
        ],
        temperature=0,
    )
