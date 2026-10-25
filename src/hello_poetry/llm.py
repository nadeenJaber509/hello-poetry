
import json
import os

from dotenv import load_dotenv
from groq import Groq

VALID_LABELS = {"positive", "negative", "neutral"}


def ask_llm(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float = 0,
) -> str:
    """Send a message to Groq and return the model's response."""

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
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
    """Extract a valid sentiment label from a JSON response."""

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
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
