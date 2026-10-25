
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
