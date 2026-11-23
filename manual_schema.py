from schemas import SupportAnswer

valid_data = {
    "answerable": True,
    "answer": "Use poetry install.",
    "commands": ["poetry install"],
    "sources": ["basic-usage.md"],
    "confidence": "high",
}

answer = SupportAnswer.model_validate(valid_data)

print("VALID DATA TEST")
print(answer)
print(answer.model_dump())
from pydantic import ValidationError

print("\nINVALID DATA TEST")

invalid_cases = [
    {
        "answerable": "yes",
        "answer": "Use poetry install.",
        "commands": ["poetry install"],
        "sources": ["basic-usage.md"],
        "confidence": "high",
    },
    {
        "answerable": True,
        "answer": "Use poetry install.",
        "commands": ["poetry install"],
        "sources": ["basic-usage.md"],
        "confidence": "very high",
    },
    {
        "answer": "Use poetry install.",
        "commands": ["poetry install"],
        "sources": ["basic-usage.md"],
        "confidence": "high",
    },
    {
        "answerable": True,
        "answer": "Use poetry install.",
        "commands": "poetry install",
        "sources": ["basic-usage.md"],
        "confidence": "high",
    },
]

for index, data in enumerate(invalid_cases, start=1):
    try:
        SupportAnswer.model_validate(data)
        print(f"Case {index}: FAILED - Invalid data was accepted")
    except ValidationError:
        print(f"Case {index}: PASSED - Validation rejected invalid data")
import schemas

print("\n========== STRUCTURED OUTPUT RETRY TEST ==========")

responses = [
    """
    {
        "answerable": true,
        "answer": "Use poetry install.",
        "commands": ["poetry install"],
        "sources": ["basic-usage.md"],
        "confidence": "very high"
    }
    """,
    """
    {
        "answerable": true,
        "answer": "Use poetry install.",
        "commands": ["poetry install"],
        "sources": ["basic-usage.md"],
        "confidence": "high"
    }
    """
]

call_count = 0

def fake_llm(**kwargs):
    global call_count

    response = responses[call_count]
    call_count += 1

    return response


schemas.ask_llm_json = fake_llm

answer, retries = schemas.get_structured_answer(
    messages=[
        {"role": "user", "content": "How do I install dependencies?"}
    ],
    model="test-model",
    max_retries=2,
)

print("Answer:", answer)
print("Retries:", retries)
print("LLM calls:", call_count)

assert answer.confidence == "high"
assert retries == 1
assert call_count == 2

print("RETRY TEST: PASSED")
print("\n========== EXHAUSTED RETRIES TEST ==========")

responses = [
    """
    {
        "answerable": true,
        "answer": "Test answer",
        "commands": [],
        "sources": [],
        "confidence": "very high"
    }
    """,
    """
    {
        "answerable": true,
        "answer": "Test answer",
        "commands": [],
        "sources": [],
        "confidence": "unknown"
    }
    """,
    """
    {
        "answerable": true,
        "answer": "Test answer",
        "commands": [],
        "sources": [],
        "confidence": "invalid"
    }
    """
]

call_count = 0

def fake_llm_always_invalid(**kwargs):
    global call_count

    response = responses[call_count]
    call_count += 1

    return response


schemas.ask_llm_json = fake_llm_always_invalid

try:
    schemas.get_structured_answer(
        messages=[
            {
                "role": "user",
                "content": "How do I install dependencies?"
            }
        ],
        model="test-model",
        max_retries=2,
    )

    print("FAILED: Invalid response was accepted")

except ValueError as error:
    print("Expected error:", error)
    print("LLM calls:", call_count)

    assert call_count == 3

    print("EXHAUSTED RETRIES TEST: PASSED")
print("\n========== MALFORMED JSON TEST ==========")

responses = [
    "This is not JSON",
    """
    {
        "answerable": true,
        "answer": "Use poetry install.",
        "commands": ["poetry install"],
        "sources": ["basic-usage.md"],
        "confidence": "high"
    }
    """
]

call_count = 0

def fake_llm_malformed(**kwargs):
    global call_count

    response = responses[call_count]
    call_count += 1

    return response


schemas.ask_llm_json = fake_llm_malformed

answer, retries = schemas.get_structured_answer(
    messages=[
        {"role": "user", "content": "How do I install dependencies?"}
    ],
    model="test-model",
    max_retries=2,
)

print("Answer:", answer)
print("Retries:", retries)
print("LLM calls:", call_count)

assert retries == 1
assert call_count == 2
assert answer.confidence == "high"

print("MALFORMED JSON TEST: PASSED")
print("\n========== MESSAGES INTEGRITY TEST ==========")

original_messages = [
    {
        "role": "user",
        "content": "How do I install dependencies?"
    }
]

original_snapshot = original_messages.copy()

responses = [
    (
        '{"answerable": true, "answer": "Invalid", '
        '"commands": [], "sources": [], "confidence": "wrong"}'
    ),
    (
        '{"answerable": true, "answer": "Use poetry install.", '
        '"commands": ["poetry install"], '
        '"sources": ["basic-usage.md"], "confidence": "high"}'
    ),
]
call_count = 0

def fake_llm_integrity(**kwargs):
    global call_count

    response = responses[call_count]
    call_count += 1

    return response


schemas.ask_llm_json = fake_llm_integrity

answer, retries = schemas.get_structured_answer(
    messages=original_messages,
    model="test-model",
    max_retries=2,
)

assert original_messages == original_snapshot
assert len(original_messages) == 1
assert retries == 1
assert call_count == 2

print("Original messages:", original_messages)
print("Retries:", retries)
print("LLM calls:", call_count)
print("MESSAGES INTEGRITY TEST: PASSED")