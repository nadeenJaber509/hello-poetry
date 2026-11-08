
"""Plain Python tool-calling agent for Poetry documentation."""

import inspect
import json
import time
from pathlib import Path
from uuid import uuid4

from hello_poetry.llm import chat_with_tools

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "traces" / "part1"


def execute_tool(name, raw_arguments, tool_functions):
    """Validate a tool request and execute an allowed function."""
    if name not in tool_functions:
        return f"Error: Unknown tool '{name}'."

    try:
        arguments = json.loads(raw_arguments or "{}")

        if not isinstance(arguments, dict):
            raise TypeError("Arguments must be a JSON object.")
        function = tool_functions[name]

        # Check that the supplied arguments match the function.
        inspect.signature(function).bind(**arguments)

        result = function(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)

    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return f"Error: Invalid tool arguments: {error}"

    except Exception as error:  # noqa: BLE001
        return f"Error: Tool execution failed: {error}"


def save_trace(trace):
    """Save the completed run without making another API call."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{uuid4().hex}.json"

    path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def run_agent(
    question,
    system_prompt,
    tool_schemas,
    tool_functions,
    model,
    max_steps=5,
):
    """Run a bounded LLM/tool loop and return its trace."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    if not question.strip():
        raise ValueError("Question must not be empty.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    steps = []
    responses = []
    total_tokens = 0
    final_answer = None
    stopped_by_limit = False

    start = time.perf_counter()

    try:
        for _ in range(max_steps):
            response = chat_with_tools(
                messages=messages,
                model=model,
                tool_schemas=tool_schemas,
            )

            responses.append(response.model_dump(mode="json"))

            if response.usage is not None:
                total_tokens += response.usage.total_tokens or 0

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            # No tool requested: the model has finished.
            if not tool_calls:
                content = (message.content or "").strip()

                if content:
                    final_answer = content
                    break

                messages.append({
                    "role": "assistant",
                    "content": (
                        "No final answer was generated. "
                        "Please provide an answer based on the "
                        "available evidence, or state that the "
                        "documentation does not support an answer."
                    ),
                })
                continue

            # Add the model's tool requests to the conversation.
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            })

            for call in tool_calls:
                tool_start = time.perf_counter()
                name = call.function.name
                arguments = call.function.arguments

                result = execute_tool(
                    name,
                    arguments,
                    tool_functions,
                )

                elapsed = time.perf_counter() - tool_start

                steps.append({
                    "tool": name,
                    "arguments": arguments,
                    "result": result[:500],
                    "seconds": round(elapsed, 3),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

        else:
            stopped_by_limit = True

    finally:
        total_seconds = round(time.perf_counter() - start, 3)

        trace = {
            "question": question,
            "final_answer": final_answer,
            "steps": steps,
            "total_tokens": total_tokens,
            "total_seconds": total_seconds,
            "model_calls": len(responses),
            "stopped_by_limit": stopped_by_limit,
            "model_responses": responses,
        }

        trace_path = save_trace(trace)

    return {
        "final_answer": final_answer,
        "steps": steps,
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "model_calls": len(responses),
        "stopped_by_limit": stopped_by_limit,
        "trace_path": str(trace_path),
    }
