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


def evidence_from_tools(initial_chunks, tool_results, max_chunks=6):
    """Keep a bounded set of passages actually retrieved or read by tools."""
    chunks = []
    seen = set()
    for chunk in initial_chunks:
        key = (chunk.get("source"), chunk.get("text"))
        if key not in seen:
            chunks.append(chunk)
            seen.add(key)
    # Keep explicitly read documents when the bounded evidence list fills up.
    ordered = sorted(tool_results, key=lambda item: item.get("tool") != "read_doc")
    for item in ordered:
        if str(item.get("result", "")).startswith("Error:"):
            continue
        try:
            value = json.loads(item["result"])
            if item["tool"] == "read_doc":
                source = json.loads(item["arguments"])["source"]
                candidates = [{"source": source, "text": value, "similarity": 0.0}]
            elif item["tool"] == "search_docs":
                candidates = value
            else:
                continue
        except KeyError, TypeError, ValueError:
            continue
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            source, passage = candidate.get("source"), candidate.get("text")
            if (
                not isinstance(source, str)
                or not isinstance(passage, str)
                or not passage
            ):
                continue
            key = (source, passage)
            if key not in seen:
                chunks.append(candidate)
                seen.add(key)
            if len(chunks) >= max_chunks:
                return chunks[:max_chunks]
    return chunks[:max_chunks]


def _researcher_read_excerpt(name, arguments, result, earlier_results):
    """Return nearby read text when a prior search already supplied its passage."""
    if name != "read_doc" or result.startswith("Error:"):
        return result
    try:
        source = json.loads(arguments)["source"]
        document = json.loads(result)
    except KeyError, TypeError, ValueError:
        return result
    if not isinstance(document, str):
        return result
    for item in earlier_results:
        if item["tool"] != "search_docs":
            continue
        try:
            chunks = json.loads(item["result"])
        except TypeError, ValueError:
            continue
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict) or chunk.get("source") != source:
                continue
            passage = chunk.get("text")
            if not isinstance(passage, str) or not passage:
                continue
            position = document.find(passage)
            if position < 0:
                continue
            start = max(0, position - 300)
            end = min(len(document), position + len(passage) + 300)
            if end - start < len(document):
                nearby = (
                    document[start:position]
                    + f"\n[Previously retrieved {source} passage omitted]\n"
                    + document[position + len(passage) : end]
                )
                return json.dumps(nearby, ensure_ascii=False)
    return result


def run_agent(
    question,
    system_prompt,
    tool_schemas,
    tool_functions,
    model,
    max_steps=5,
    compact_read_doc=False,
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
    tool_results = []
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

                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "No final answer was generated. "
                            "Please provide an answer based on the "
                            "available evidence, or state that the "
                            "documentation does not support an answer."
                        ),
                    }
                )
                continue

            # Add the model's tool requests to the conversation.
            messages.append(
                {
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
                }
            )

            for call in tool_calls:
                tool_start = time.perf_counter()
                name = call.function.name
                arguments = call.function.arguments

                result = execute_tool(
                    name,
                    arguments,
                    tool_functions,
                )
                message_result = (
                    _researcher_read_excerpt(name, arguments, result, tool_results)
                    if compact_read_doc
                    else result
                )
                tool_results.append(
                    {"tool": name, "arguments": arguments, "result": result}
                )

                elapsed = time.perf_counter() - tool_start

                steps.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "result": result[:500],
                        "seconds": round(elapsed, 3),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": message_result,
                    }
                )

        else:
            stopped_by_limit = True

    finally:
        total_seconds = round(time.perf_counter() - start, 3)

        trace = {
            "question": question,
            "final_answer": final_answer,
            "steps": steps,
            "tool_results": tool_results,
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
        "tool_results": tool_results,
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "model_calls": len(responses),
        "stopped_by_limit": stopped_by_limit,
        "trace_path": str(trace_path),
    }
