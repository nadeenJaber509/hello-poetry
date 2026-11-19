import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import BadRequestError, Groq

VALID_LABELS = {"positive", "negative", "neutral"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env() -> None:
    """Load the project-root .env without replacing configured environment values."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def ask_llm(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float = 0,
) -> str:
    """Send a message to Groq and return the model's response."""
    answer, _ = ask_llm_with_usage(system_prompt, user_message, model, temperature)
    return answer


def ask_llm_with_usage(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float = 0,
) -> tuple[str, int]:
    """Return response text and token count for evaluation."""

    load_project_env()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    client = Groq(api_key=api_key, max_retries=0)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )

    tokens = response.usage.total_tokens if response.usage else 0
    return response.choices[0].message.content or "", tokens


def parse_label(text: str) -> str:
    """Extract a valid sentiment label from a JSON response."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError, TypeError:
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


def chat_with_tools(messages, model, tool_schemas):
    """Call Groq and let the model request local tools."""
    load_project_env()
    client = Groq(max_retries=0)
    for attempt in range(2):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=0,
            )
        except BadRequestError as error:
            details = (
                error.body.get("error", {}) if isinstance(error.body, dict) else {}
            )
            if (
                attempt
                or not isinstance(details, dict)
                or details.get("code") != "tool_use_failed"
                or "'read_doc<|channel|>commentary'" not in details.get("message", "")
            ):
                raise


def chat_json(messages, model, json_mode=True):
    """Call Groq with optional JSON mode."""
    load_project_env()
    client = Groq(max_retries=0)

    options = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    if json_mode:
        options["response_format"] = {"type": "json_object"}

    return client.chat.completions.create(**options)
