
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

VALID_LABELS = {"positive", "negative", "neutral"}


def ask_llm(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float = 0,
) -> str:
    """Send a system prompt and user message to Groq."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Check your .env file."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


def parse_label(text: str) -> str:
    """Parse a sentiment label from a JSON response."""
    if not isinstance(text, str) or not text.strip():
        return "invalid"

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return "invalid"

    if not isinstance(data, dict):
        return "invalid"

    label = data.get("label")
    if not isinstance(label, str):
        return "invalid"

    label = label.strip().lower()

    if label in VALID_LABELS:
        return label

    return "invalid"


def ask_llm_with_tools(
    messages: list[dict],
    tools: list[dict] | None,
    model: str,
    temperature: float = 0,
    tool_choice: str | None = "auto",
    response_format: dict | None = None,

) -> dict:
    """Send messages and optional tools to Groq."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Check your .env file."
        )

    client = Groq(api_key=api_key)

    request_args = {
    "model": model,
    "messages": messages,
    "temperature": temperature,
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
}
    if response_format is not None:
        request_args["response_format"] = response_format
    if tools is not None and len(tools) > 0:
        request_args["tools"] = tools
        request_args["tool_choice"] = tool_choice or "auto"

    response = client.chat.completions.create(
        **request_args
    )

    message = response.choices[0].message

    return {
        "message": message.model_dump(exclude_none=True),
        "usage": (
            response.usage.model_dump()
            if response.usage else {}
        ),
    }
def ask_llm_json(
    messages: list[dict],
    model: str,
    temperature: float = 0,
) -> str:
    """Send messages to Groq and request a JSON response."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Check your .env file."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content or ""