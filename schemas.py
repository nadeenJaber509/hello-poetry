from typing import Literal

from pydantic import BaseModel, StrictBool, ValidationError

from task6.llm import ask_llm_json


class SupportAnswer(BaseModel):
    answerable: StrictBool
    answer: str
    commands: list[str]
    sources: list[str]
    confidence: Literal["low", "medium", "high"]

def get_structured_answer(
    messages: list[dict],
    model: str,
    max_retries: int = 2,
) -> tuple[SupportAnswer, int]:
    """Get and validate a structured answer from the LLM."""

    conversation = messages.copy()

    for attempt in range(max_retries + 1):
        raw_response = ask_llm_json(
            messages=conversation,
            model=model,
            temperature=0,
        )

        try:
            answer = SupportAnswer.model_validate_json(
                raw_response
            )

            return answer, attempt

        except ValidationError as error:
            conversation.append({
                "role": "assistant",
                "content": raw_response,
            })

            conversation.append({
                "role": "user",
                "content": (
                    "Your previous response did not match "
                    "the required SupportAnswer structure. "
                    "Please correct the JSON response. "
                    f"Validation errors: {error}"
                ),
            })

    raise ValueError(
        f"Could not get a valid SupportAnswer "
        f"after {max_retries + 1} attempts."
    )