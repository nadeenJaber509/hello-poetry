
"""Run the remaining Part 1.2 agent tests."""

import json
from pathlib import Path

from hello_poetry.agent import run_agent
from hello_poetry.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "results" / "final_task_part1_tests.json"

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are a Poetry documentation assistant. "
    "Use the available tools to find evidence before answering. "
    "Answer only using the Poetry documentation. "
    "If the docs do not support an answer, say so."
)

QUESTIONS = [
    "How do I create a new Poetry project?",
    "What is the purpose of poetry.lock?",
    "How can I check which dependencies are outdated?",
    "Who won the FIFA World Cup in 2022?",
]


def main():
    """Run each question once and save its result."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        results = json.loads(
            OUTPUT_FILE.read_text(encoding="utf-8")
        )
    else:
        results = {}

    for question in QUESTIONS:
        if question in results:
            print(f"\nSKIPPING saved question: {question}")
            continue

        print(f"\nQUESTION: {question}")

        try:
            result = run_agent(
                question=question,
                system_prompt=SYSTEM_PROMPT,
                tool_schemas=TOOL_SCHEMAS,
                tool_functions=TOOL_FUNCTIONS,
                model=MODEL,
                max_steps=5,
            )

            results[question] = result

        except Exception as error:  # noqa: BLE001
            print(f"ERROR: {error}")
            # Stop without repeating an unsuccessful API call.
            break

        finally:
            OUTPUT_FILE.write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print("ANSWER:", result["final_answer"])
        print("MODEL CALLS:", result["model_calls"])
        print("TOKENS:", result["total_tokens"])
        print("STOPPED BY LIMIT:", result["stopped_by_limit"])

        print("TOOLS:")
        for step in result["steps"]:
            print(
                f"  {step['tool']} "
                f"{step['arguments']}"
            )

        print("TRACE:", result["trace_path"])


if __name__ == "__main__":
    main()
