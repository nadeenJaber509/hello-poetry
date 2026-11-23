
import json
from unittest.mock import patch

from agent import (
    FALLBACK_ANSWER,
    run_agent,
)

MODEL = "test-model"
SYSTEM_PROMPT = "You are a Poetry documentation assistant."


def make_response(content="", tool_calls=None, tokens=10):
    return {
        "message": {
            "content": content,
            "tool_calls": tool_calls or [],
        },
        "usage": {
            "total_tokens": tokens,
        },
    }


def make_tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def test_direct_answer_without_tools():
    response = make_response(
        content="Use poetry install.",
        tokens=15,
    )

    with patch(
        "agent.ask_llm_with_tools",
        return_value=response,
    ) as mock_llm:

        result = run_agent(
            question="How do I install dependencies?",
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=[],
            tool_functions={},
            model=MODEL,
        )

    assert result["final_answer"] == "Use poetry install."
    assert result["steps"] == []
    assert result["total_tokens"] == 15
    assert result["stopped_by_limit"] is False
    assert mock_llm.call_count == 1


def test_successful_tool_execution():
    tool_call = make_tool_call(
        "search_docs",
        {"query": "How to install dependencies"},
    )

    responses = [
        make_response(tool_calls=[tool_call], tokens=10),
        make_response(
            content="Use poetry install.",
            tokens=15,
        ),
    ]

    def fake_search(query):
        return {
            "source": "basic-usage.md",
            "text": "Run poetry install to install dependencies.",
        }

    with patch(
        "agent.ask_llm_with_tools",
        side_effect=responses,
    ) as mock_llm:

        result = run_agent(
            question="How do I install dependencies?",
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=[{"type": "function"}],
            tool_functions={"search_docs": fake_search},
            model=MODEL,
        )

    assert result["final_answer"] == "Use poetry install."
    assert len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "search_docs"
    assert "basic-usage.md" in result["steps"][0]["result"]
    assert result["total_tokens"] == 25
    assert mock_llm.call_count == 2


def test_tool_execution_error():
    tool_call = make_tool_call(
        "search_docs",
        {"query": "installation"},
    )

    responses = [
        make_response(tool_calls=[tool_call]),
        make_response(content="The search failed."),
    ]

    def failing_search(query):
        raise RuntimeError("Test search failure")

    with patch(
        "agent.ask_llm_with_tools",
        side_effect=responses,
    ):

        result = run_agent(
            question="How do I install dependencies?",
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=[{"type": "function"}],
            tool_functions={"search_docs": failing_search},
            model=MODEL,
        )

    assert len(result["steps"]) == 1
    assert "Test search failure" in result["steps"][0]["result"]
    assert result["final_answer"] == "The search failed."


def test_budget_exhaustion():
    tool_call = make_tool_call(
        "search_docs",
        {"query": "installation"},
    )

    budget = {
        "max_calls": 1,
        "used_calls": 0,
    }

    response = make_response(tool_calls=[tool_call])

    with patch(
        "agent.ask_llm_with_tools",
        return_value=response,
    ) as mock_llm:

        result = run_agent(
            question="How do I install dependencies?",
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=[{"type": "function"}],
            tool_functions={
                "search_docs": lambda query: {"text": "Install using Poetry."}
            },
            model=MODEL,
            call_budget=budget,
        )

    assert budget["used_calls"] == 1
    assert mock_llm.call_count == 1
    assert result["final_answer"] == FALLBACK_ANSWER
    assert result["stopped_by_limit"] is True


def test_max_steps_forces_final_answer():
    tool_call = make_tool_call(
        "search_docs",
        {"query": "installation"},
    )

    responses = [
        make_response(tool_calls=[tool_call]),
        make_response(
            content="Use poetry install.",
            tokens=15,
        ),
    ]

    with patch(
        "agent.ask_llm_with_tools",
        side_effect=responses,
    ) as mock_llm:

        result = run_agent(
            question="How do I install dependencies?",
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=[{"type": "function"}],
            tool_functions={
                "search_docs": lambda query: {"text": "Use poetry install."}
            },
            model=MODEL,
            max_steps=1,
        )

    assert result["final_answer"] == "Use poetry install."
    assert len(result["steps"]) == 1
    assert result["stopped_by_limit"] is True
    assert mock_llm.call_count == 2
