from schemas import get_structured_answer

MODEL = "openai/gpt-oss-20b"

messages = [
    {
        "role": "system",
        "content": """
        You are a Poetry documentation support assistant.
        Return a JSON object with exactly these fields:
        answerable, answer, commands, sources, confidence.

        confidence must be low, medium, or high.
        commands and sources must be lists of strings.
        If you do not know the answer from the provided context,
        set answerable to false.
        """
    },
    {
        "role": "user",
        "content": (
            "According to Poetry documentation, "
            "how can I add a package to my project?"
        )
    }
]

result, retries = get_structured_answer(
    messages=messages,
    model=MODEL,
)

print("\n========== STRUCTURED ANSWER ==========")
print(result.model_dump_json(indent=2))

print("\nRetries:", retries)