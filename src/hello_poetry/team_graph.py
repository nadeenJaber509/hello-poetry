"""Optional Part 8 LangGraph version of the existing team flow."""

import json
import time
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from hello_poetry.agents import run_researcher, run_reviewer, run_router, run_writer
from hello_poetry.guardrails import (
    REFUSAL,
    check_evidence,
    verify_citations,
    verify_commands,
)

TRACE_DIR = Path(__file__).resolve().parents[2] / "traces" / "graph"


class TeamState(TypedDict, total=False):
    question: str
    model: str
    trace: list[dict]
    model_calls: int
    total_tokens: int
    max_model_calls: int
    max_review_rounds: int
    remaining_calls: int
    review_rounds: int
    route: dict
    chunks: list[dict]
    evidence: list[dict]
    evidence_chunks: list[dict]
    research: dict
    support_answer: object
    review: dict
    review_problems: list[str]
    json_ok_first_try: bool


def build_graph(agent_functions=None):
    """Use the existing four agents as graph nodes with conditional routes."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install langgraph for the Part 8 experiment") from exc

    functions = agent_functions or {
        "router": run_router,
        "researcher": run_researcher,
        "writer": run_writer,
        "reviewer": run_reviewer,
    }

    def node(name):
        def call(state):
            available = state["max_model_calls"] - state["model_calls"]
            if available < 1:
                raise RuntimeError("max_model_calls exceeded")
            prepared = {**state, "remaining_calls": available}
            result = functions[name](prepared)
            if result["model_calls"] > state["max_model_calls"]:
                raise RuntimeError("max_model_calls exceeded")
            if name == "reviewer":
                result["review_rounds"] = state["review_rounds"] + 1
                if not result["review"]["approved"]:
                    result["review_problems"] = result["review"]["problems"]
            return result

        return call

    def after_router(state):
        return END if state["route"]["category"] == "out_of_scope" else "researcher"

    def after_researcher(state):
        return "writer" if check_evidence(state["chunks"]).has_evidence else END

    def after_reviewer(state):
        if (
            state["review"]["approved"]
            or state["review_rounds"] >= state["max_review_rounds"]
        ):
            return END
        return "writer"

    builder = StateGraph(TeamState)
    for name in ("router", "researcher", "writer", "reviewer"):
        builder.add_node(name, node(name))
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", after_router)
    builder.add_conditional_edges("researcher", after_researcher)
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges("reviewer", after_reviewer)
    return builder.compile()


def answer_with_graph(
    question: str,
    max_review_rounds: int = 2,
    max_model_calls: int = 12,
    model: str = "openai/gpt-oss-20b",
    agent_functions=None,
) -> dict:
    """Run the graph and apply the same final Python checks as the coordinator."""
    if not question.strip() or max_review_rounds < 1 or max_model_calls < 1:
        raise ValueError("Question and limits must be valid")
    started = time.perf_counter()
    state = build_graph(agent_functions).invoke(
        {
            "question": question,
            "model": model,
            "trace": [],
            "model_calls": 0,
            "total_tokens": 0,
            "review_rounds": 0,
            "max_model_calls": max_model_calls,
            "max_review_rounds": max_review_rounds,
        },
        {"recursion_limit": 16},
    )
    answer = state.get("support_answer")
    blocked_by = None
    if state["route"]["category"] == "out_of_scope":
        blocked_by = "scope"
    elif not check_evidence(state["chunks"]).has_evidence:
        blocked_by = "evidence"
    elif not state["review"]["approved"]:
        blocked_by = "reviewer"
    elif not answer.answerable or not answer.answer.strip():
        blocked_by = "evidence"
    elif verify_citations(answer, [item["source"] for item in state["evidence"]]):
        blocked_by = "citations"
    elif verify_commands(answer, [item["quote"] for item in state["evidence"]]):
        blocked_by = "commands"
    result = {
        "answer": REFUSAL if blocked_by else answer.answer,
        "answerable": not bool(blocked_by),
        "blocked_by": blocked_by,
        "commands": [] if blocked_by else answer.commands,
        "sources": [] if blocked_by else answer.sources,
        "confidence": "low" if blocked_by else answer.confidence,
        "trace": state["trace"],
        "model_calls": state["model_calls"],
        "total_tokens": state["total_tokens"],
        "total_seconds": time.perf_counter() - started,
        "review_rounds": state["review_rounds"],
        "json_ok_first_try": state.get("json_ok_first_try"),
    }
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{uuid4().hex}.json"
    path.write_text(
        json.dumps({"question": question, **result}, indent=2, default=str),
        encoding="utf-8",
    )
    result["trace_path"] = str(path)
    return result
