"""Four small agents sharing a plain state dictionary."""

import json
import re
import time

from hello_poetry.agent import evidence_from_tools, run_agent
from hello_poetry.guardrails import EVIDENCE_THRESHOLD
from hello_poetry.llm import chat_json
from hello_poetry.prompts import (
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    ROUTER_PROMPT,
    WRITER_PROMPT,
)
from hello_poetry.schemas import get_structured_answer
from hello_poetry.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, search_docs


def select_research_chunks(question: str, chunks: list[dict], k: int = 3) -> list[dict]:
    """Prefer retrieved examples whose Poetry command appears in the question."""
    words = set(re.findall(r"[a-z][a-z0-9-]*", question.casefold())) - {
        "a",
        "about",
        "an",
        "and",
        "can",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "my",
        "of",
        "on",
        "the",
        "to",
        "using",
        "what",
        "when",
        "where",
        "with",
    }

    def rank(chunk: dict) -> tuple[bool, float]:
        text = chunk["text"].casefold()
        examples = re.findall(
            r"```(?:bash|sh|shell|console)?\n(.*?)```", text, flags=re.DOTALL
        )
        commands = re.findall(r"`poetry\s+([a-z][a-z0-9-]*)\b", text)
        for example in examples:
            commands.extend(
                re.findall(r"(?m)^\s*poetry\s+([a-z][a-z0-9-]*)\b", example)
            )
        return bool(words.intersection(commands)), chunk["similarity"]

    eligible = [chunk for chunk in chunks if chunk["similarity"] >= EVIDENCE_THRESHOLD]
    return sorted(eligible, key=rank, reverse=True)[:k]


def _record(state, name, input_text, output, calls=1, tokens=0, seconds=0):
    state["trace"].append(
        {
            "agent": name,
            "input": str(input_text)[:500],
            "output": str(output)[:500],
            "full_output": output,
            "model_calls": calls,
            "tokens": tokens,
            "seconds": seconds,
        }
    )
    state["model_calls"] += calls
    state["total_tokens"] += tokens
    return state


def _json_call(prompt, content, model):
    response = chat_json(
        [{"role": "system", "content": prompt}, {"role": "user", "content": content}],
        model,
    )
    data = json.loads(response.choices[0].message.content)
    tokens = response.usage.total_tokens if response.usage else 0
    return data, tokens


def run_router(state):
    state = dict(state)
    start = time.perf_counter()
    data, tokens = _json_call(ROUTER_PROMPT, state["question"], state["model"])
    if data.get("category") not in {
        "how_to",
        "concept",
        "troubleshooting",
        "out_of_scope",
    }:
        raise ValueError("Router returned an invalid category")
    state["route"] = data
    return _record(
        state,
        "router",
        state["question"],
        data,
        tokens=tokens,
        seconds=time.perf_counter() - start,
    )


def run_researcher(state):
    state = dict(state)
    start = time.perf_counter()
    chunks = select_research_chunks(
        state["question"], search_docs(state["question"], k=10)
    )
    state["chunks"] = chunks
    # The bounded tool loop is retained for the Researcher; the directly retrieved
    # chunks remain the auditable evidence passed to the Writer.
    research = run_agent(
        state["question"],
        RESEARCHER_PROMPT,
        TOOL_SCHEMAS,
        TOOL_FUNCTIONS,
        state["model"],
        max_steps=min(3, state.get("remaining_calls", 3)),
        compact_read_doc=True,
    )
    evidence_chunks = evidence_from_tools(chunks, research.get("tool_results", []))
    state["evidence_chunks"] = evidence_chunks
    verified = [
        {"source": c["source"], "quote": c["text"][:1200]} for c in evidence_chunks
    ]
    try:
        draft = (research["final_answer"] or "").strip()
        if draft.startswith("```") and draft.endswith("```"):
            draft = draft.partition("\n")[2].removesuffix("```").strip()
        proposed = json.loads(draft)
        items = proposed["evidence"] if isinstance(proposed, dict) else proposed
        if not isinstance(items, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("source"), str)
            and isinstance(item.get("quote"), str)
            and any(
                item["source"] == c["source"] and item["quote"] in c["text"]
                for c in evidence_chunks
            )
            for item in items
        ):
            raise ValueError("Researcher evidence was not supported")
        state["evidence"] = items or verified
    except ValueError, KeyError, TypeError:
        state["evidence"] = verified
    state["research"] = research
    recorded = _record(
        state,
        "researcher",
        state["question"],
        state["evidence"],
        calls=research["model_calls"],
        tokens=research["total_tokens"],
        seconds=time.perf_counter() - start,
    )
    recorded["trace"][-1]["agent_trace_path"] = research["trace_path"]
    return recorded


def run_writer(state):
    state = dict(state)
    start = time.perf_counter()
    content = json.dumps(
        {
            "question": state["question"],
            "evidence": state["evidence"],
            "review_problems": state.get("review_problems", []),
        }
    )
    answer, retries = get_structured_answer(
        [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": content},
        ],
        state["model"],
        max_retries=min(2, state.get("remaining_calls", 3) - 1),
    )
    state["support_answer"] = answer
    state["json_ok_first_try"] = retries == 0
    return _record(
        state,
        "writer",
        content,
        answer.model_dump(),
        calls=retries + 1,
        tokens=answer._total_tokens,
        seconds=time.perf_counter() - start,
    )


def run_reviewer(state):
    state = dict(state)
    start = time.perf_counter()
    content = json.dumps(
        {
            "question": state["question"],
            "evidence": state["evidence"],
            "answer": state["support_answer"].model_dump(),
        }
    )
    data, tokens = _json_call(REVIEWER_PROMPT, content, state["model"])
    if type(data.get("approved")) is not bool or not isinstance(
        data.get("problems"), list
    ):
        raise ValueError("Reviewer returned invalid JSON fields")
    state["review"] = data
    return _record(
        state,
        "reviewer",
        content,
        data,
        tokens=tokens,
        seconds=time.perf_counter() - start,
    )
