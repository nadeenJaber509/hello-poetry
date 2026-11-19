"""Structured answer schema and JSON validation for Poetry support."""

import json
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, PrivateAttr, StrictBool, ValidationError

from hello_poetry.llm import chat_json

# --------------------------------------------------
# 1. Structured answer schema
# --------------------------------------------------


class SupportAnswer(BaseModel):
    """Validated answer returned by the Poetry support system."""

    model_config = ConfigDict(extra="forbid")

    answerable: StrictBool
    answer: str
    commands: list[str]
    sources: list[str]
    confidence: Literal["low", "medium", "high"]
    _total_tokens: int = PrivateAttr(default=0)


# --------------------------------------------------
# 2. Directory for saved model responses
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results" / "final_task" / "structured_runs"


# --------------------------------------------------
# 3. Structured answer with validation and retries
# --------------------------------------------------


def get_structured_answer(
    messages,
    model,
    max_retries=2,
    json_mode=True,
):
    """Get a validated JSON answer, retrying invalid responses."""

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    # Copy messages to avoid changing the original conversation.
    conversation = [dict(message) for message in messages]

    # Tell the model exactly which JSON fields to return.
    conversation.insert(
        0,
        {
            "role": "system",
            "content": (
                "Return one valid JSON object with exactly these fields: "
                "answerable (boolean), "
                "answer (string), "
                "commands (array of strings), "
                "sources (array of strings), "
                "confidence (low, medium, or high). "
                "Do not include any text outside the JSON object."
            ),
        },
    )

    # Prepare a unique file for this run.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_path = RESULTS_DIR / f"{uuid4().hex}.jsonl"
    total_tokens = 0

    # Initial attempt + max_retries additional attempts.
    for attempt in range(max_retries + 1):
        # Call Groq using the selected mode.
        response = chat_json(
            conversation,
            model,
            json_mode=json_mode,
        )
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0)
        if isinstance(tokens, int):
            total_tokens += tokens

        content = response.choices[0].message.content or ""

        # Save every model response before validation.
        record = {
            "attempt": attempt + 1,
            "timestamp": time.time(),
            "model": model,
            "json_mode": json_mode,
            "content": content,
        }

        with run_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Validate the response using Pydantic.
        try:
            answer = SupportAnswer.model_validate_json(content)
            answer._total_tokens = total_tokens

            # Return validated answer and retries used.
            return answer, attempt

        except ValidationError as error:
            # No more retries available.
            if attempt == max_retries:
                raise ValueError(
                    "Invalid structured answer after "
                    f"{max_retries + 1} attempts: {error}"
                ) from error

            # Add the invalid response to the conversation.
            conversation.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            # Ask the model to correct its JSON response.
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed schema validation. "
                        f"Validation error: {error}. "
                        "Return a corrected JSON object with all required "
                        "fields and valid types."
                    ),
                }
            )

    raise RuntimeError("Unexpected end of structured answer loop.")
