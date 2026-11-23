
from agent import run_agent
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, read_doc

print("\n========== READ_DOC TEST ==========")

result = read_doc(source="basic-usage.md")
print(result)
MODEL = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """
You are a helpful Poetry documentation assistant.

Your task is to answer questions using the available Poetry documentation tools.

Follow these rules:

1. Use search_docs to search the documentation when relevant.
2. Start with one search using the user's question.
3. Do not repeat the same search or search with minor wording changes.
4. If the results are relevant, use them to answer the question.
5. Use read_doc only when you need more details from a specific document.
6. Do not call tools repeatedly when they do not provide useful information.
7. If the documentation does not contain the answer, stop searching
   and clearly say that the information could not be found.
8. Do not invent information or use unsupported assumptions.
9. If the question is outside the scope of Poetry documentation,
   explain that the available documentation does not cover the topic.

Keep your answer clear, concise, and easy to understand.
Always provide a final answer instead of continuing to search unnecessarily.
"""


if __name__ == "__main__":

    question = (
    "Read basic-usage.md starting from character 6000. "
    "Use read_doc with start=6000 and max_chars=2000. "
    "Explain how poetry run works using the retrieved content."
)
    result = run_agent(
        question=question,
        system_prompt=SYSTEM_PROMPT,
        tool_schemas=TOOL_SCHEMAS,
        tool_functions=TOOL_FUNCTIONS,
        model=MODEL,
        max_steps=5,
    )

    print("\n========== FINAL ANSWER ==========")
    print(result["final_answer"])

    print("\n========== AGENT STEPS ==========")

    for index, step in enumerate(result["steps"], start=1):
        print(f"\nStep {index}")
        print("Tool:", step["tool"])
        print("Arguments:", step["args"])
        print("Result:", step["result"])
        print("Time:", step["seconds"], "seconds")

    print("\n========== USAGE ==========")
    print("Total tokens:", result["total_tokens"])
    print("Stopped by limit:", result["stopped_by_limit"])
print("\n========== UNKNOWN TOOL TEST ==========")

FAKE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fake_tool",
            "description": "A tool that does not exist.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]

fake_result = run_agent(
    question="Use fake_tool to find information about Poetry.",
    system_prompt=SYSTEM_PROMPT,
    tool_schemas=FAKE_TOOL_SCHEMAS,
    tool_functions=TOOL_FUNCTIONS,
    model=MODEL,
    max_steps=3,
)

print("Final answer:", fake_result["final_answer"])
print("Steps:", fake_result["steps"])
print("Stopped by limit:", fake_result["stopped_by_limit"])