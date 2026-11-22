"""Plain Python coordinator for the Poetry support agents."""

import json
import time
from pathlib import Path
from uuid import uuid4

from hello_poetry.agents import run_researcher, run_reviewer, run_router, run_writer
from hello_poetry.guardrails import (
    REFUSAL,
    check_evidence,
    verify_citations,
    verify_commands,
)

TRACE_DIR = Path(__file__).resolve().parents[2] / "traces" / "team"


def answer_with_team(
    question,
    max_review_rounds=2,
    max_model_calls=12,
    model="openai/gpt-oss-20b",
    use_reviewer=True,
    agent_functions=None,
):
    """Route, research, draft, review, and verify with a hard call budget."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    if (
        max_review_rounds < 0
        or (use_reviewer and max_review_rounds < 1)
        or max_model_calls < 1
    ):
        raise ValueError("Limits must allow a review round and a model call")
    start = time.perf_counter()
    state = {
        "question": question,
        "model": model,
        "trace": [],
        "model_calls": 0,
        "total_tokens": 0,
    }
    rounds = 0
    functions = agent_functions or {
        "router": run_router,
        "researcher": run_researcher,
        "writer": run_writer,
        "reviewer": run_reviewer,
    }
    blocked_by = None
    answer = None
    try:
        state["remaining_calls"] = max_model_calls - state["model_calls"]
        state = functions["router"](state)
        if state["model_calls"] > max_model_calls:
            raise RuntimeError("max_model_calls exceeded")
        if state["route"]["category"] == "out_of_scope":
            blocked_by = "scope"
        else:
            state["remaining_calls"] = max_model_calls - state["model_calls"]
            if state["remaining_calls"] < 1:
                raise RuntimeError("max_model_calls exceeded")
            state = functions["researcher"](state)
            if state["model_calls"] > max_model_calls:
                raise RuntimeError("max_model_calls exceeded")
            if not check_evidence(state["chunks"]).has_evidence:
                blocked_by = "evidence"
            else:
                attempts = max_review_rounds if use_reviewer else 1
                for attempt in range(attempts):
                    state["remaining_calls"] = max_model_calls - state["model_calls"]
                    if state["remaining_calls"] < 1:
                        raise RuntimeError("max_model_calls exceeded")
                    state = functions["writer"](state)
                    if state["model_calls"] > max_model_calls:
                        raise RuntimeError("max_model_calls exceeded")
                    if use_reviewer:
                        if state["model_calls"] >= max_model_calls:
                            raise RuntimeError("max_model_calls exceeded")
                        state = functions["reviewer"](state)
                        rounds += 1
                        if state["model_calls"] > max_model_calls:
                            raise RuntimeError("max_model_calls exceeded")
                        if not state["review"]["approved"]:
                            if attempt == attempts - 1:
                                blocked_by = "reviewer"
                                break
                            state["review_problems"] = state["review"]["problems"]
                            continue
                    answer = state["support_answer"]
                    if not answer.answerable or not answer.answer.strip():
                        blocked_by = "evidence"
                    elif verify_citations(
                        answer,
                        [item["source"] for item in state["evidence"]],
                    ):
                        blocked_by = "citations"
                    elif verify_commands(
                        answer,
                        [item["quote"] for item in state["evidence"]],
                    ):
                        blocked_by = "commands"
                    break
    finally:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{uuid4().hex}.json"
        path.write_text(
            json.dumps(
                {
                    "question": question,
                    "trace": state["trace"],
                    "model_calls": state["model_calls"],
                    "blocked_by": blocked_by,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return {
        "answer": REFUSAL if blocked_by else answer.answer,
        "answerable": not bool(blocked_by),
        "blocked_by": blocked_by,
        "commands": [] if blocked_by else answer.commands,
        "sources": [] if blocked_by else answer.sources,
        "confidence": "low" if blocked_by else answer.confidence,
        "trace": state["trace"],
        "trace_path": str(path),
        "total_tokens": state["total_tokens"],
        "model_calls": state["model_calls"],
        "total_seconds": time.perf_counter() - start,
        "review_rounds": rounds,
        "json_ok_first_try": state.get("json_ok_first_try"),
    }
