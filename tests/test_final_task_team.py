"""Offline checks for the final-task guardrails and coordinator."""

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from groq import APIConnectionError, BadRequestError, RateLimitError
from streamlit.testing.v1 import AppTest

from hello_poetry import coordinator, llm
from hello_poetry.agent import _researcher_read_excerpt, execute_tool, run_agent
from hello_poetry.agents import run_researcher, select_research_chunks
from hello_poetry.guardrails import (
    REFUSAL,
    guarded_answer,
    verify_citations,
    verify_commands,
)
from hello_poetry.schemas import SupportAnswer
from scripts import final_task_evaluate as evaluation
from scripts.final_task_summarize import manual_grade_ids


def test_python_output_checks():
    answer = SupportAnswer(
        answerable=True,
        answer="x",
        commands=["Poetry  Add   flask", "poetry sparkle"],
        sources=["cli.md", "fake.md"],
        confidence="medium",
    )
    assert verify_citations(answer, ["cli.md"]) == ["fake.md"]
    assert verify_commands(answer, ["Run poetry add flask here."]) == ["poetry sparkle"]


def test_global_cli_flag_commands_require_documented_option_and_subcommand():
    answer = SupportAnswer(
        answerable=True,
        answer="Use a global option.",
        commands=[
            "poetry install --no-plugins",
            "poetry --no-plugins install",
            "poetry --no-plugins",
            "poetry teleport",
            "poetry teleport --no-plugins",
            "poetry install --made-up-flag",
        ],
        sources=["cli.md"],
        confidence="high",
    )
    evidence = ["* `--no-plugins`: Disables plugins."]
    assert verify_commands(answer, evidence) == [
        "poetry teleport",
        "poetry teleport --no-plugins",
        "poetry install --made-up-flag",
    ]
    assert verify_commands(answer, ["Plugins can be disabled."]) == answer.commands


def test_documented_command_examples_accept_variable_names_and_options():
    evidence = [
        "## Project setup\n```bash\npoetry new poetry-demo\n```",
        (
            "## new\n```bash\npoetry new my-package\n```\n"
            "```bash\npoetry new --flat my-package\n```\n"
            "```bash\npoetry new my-folder --name my-package\n```"
        ),
    ]
    answer = SupportAnswer(
        answerable=True,
        answer="Create a project.",
        commands=[
            "poetry new <project-name>",
            "poetry new demo",
            "poetry new --flat <project-name>",
            "poetry new <folder> --name <project-name>",
        ],
        sources=["basic-usage.md", "cli.md"],
        confidence="high",
    )
    assert verify_commands(answer, evidence) == []


def test_command_examples_reject_invented_verbs_flags_and_shapes():
    evidence = [
        (
            "## new\n```bash\npoetry new my-package\n```\n"
            "```bash\npoetry new --flat my-package\n```"
        ),
        "### source add\n```bash\npoetry source add demo https://example.com\n```",
        "The fictional example poetry new --warp <project-name> is unsupported.",
        "```bash\npoetry teleport demo\n```",
    ]
    answer = SupportAnswer(
        answerable=True,
        answer="Unsupported examples.",
        commands=[
            "poetry teleport <project-name>",
            "poetry new --warp <project-name>",
            "poetry new --flat",
            "poetry new <folder> --name <project-name>",
            "poetry new <project-name>; echo unsafe",
            "poetry source teleport demo https://example.com",
        ],
        sources=["cli.md"],
        confidence="low",
    )
    assert verify_commands(answer, evidence) == answer.commands


@pytest.mark.parametrize(
    "snippet",
    ["```bash\n$ poetry add pendulum\n```", "```bash\n$ poetry add pendulum\n"],
)
def test_shell_prompt_example_supports_variable_dependency_but_not_invented_commands(
    snippet,
):
    answer = SupportAnswer(
        answerable=True,
        answer="Use the documented command.",
        commands=[
            "poetry add requests",
            "poetry teleport requests",
            "poetry add --imaginary requests",
        ],
        sources=["basic-usage.md"],
        confidence="high",
    )
    assert verify_commands(answer, [snippet]) == answer.commands[1:]


def _router(state):
    state["route"] = {"category": "how_to"}
    state["model_calls"] += 1
    state["trace"].append({"agent": "router"})
    return state


def _researcher(state):
    state["chunks"] = [
        {"source": "cli.md", "text": "Use poetry add flask", "similarity": 0.8}
    ]
    state["evidence"] = [{"source": "cli.md", "quote": "Use poetry add flask"}]
    state["model_calls"] += 1
    state["trace"].append({"agent": "researcher"})
    return state


def _writer(state):
    state["support_answer"] = SupportAnswer(
        answerable=True,
        answer="Use poetry add flask",
        commands=["poetry add flask"],
        sources=["cli.md"],
        confidence="high",
    )
    state["model_calls"] += 1
    state["trace"].append({"agent": "writer"})
    return state


def test_reviewer_rejects_then_approves(tmp_path):
    count = 0

    def reviewer(state):
        nonlocal count
        count += 1
        state["review"] = {"approved": count == 2, "problems": ["Fix claim"]}
        state["model_calls"] += 1
        state["trace"].append({"agent": "reviewer"})
        return state

    with (
        patch.object(coordinator, "TRACE_DIR", tmp_path),
        patch.object(coordinator, "run_router", _router),
        patch.object(coordinator, "run_researcher", _researcher),
        patch.object(coordinator, "run_writer", _writer),
        patch.object(coordinator, "run_reviewer", reviewer),
    ):
        result = coordinator.answer_with_team("How?", max_review_rounds=2)
    assert result["answer"] == "Use poetry add flask"
    assert result["review_rounds"] == 2
    assert result["model_calls"] == 6


def test_reviewer_rejects_every_time(tmp_path):
    def reject(state):
        state["review"] = {"approved": False, "problems": ["Unsupported"]}
        state["model_calls"] += 1
        return state

    with (
        patch.object(coordinator, "TRACE_DIR", tmp_path),
        patch.object(coordinator, "run_router", _router),
        patch.object(coordinator, "run_researcher", _researcher),
        patch.object(coordinator, "run_writer", _writer),
        patch.object(coordinator, "run_reviewer", reject),
    ):
        result = coordinator.answer_with_team("How?", max_review_rounds=2)
    assert result["answer"] == REFUSAL
    assert result["blocked_by"] == "reviewer"


@pytest.mark.parametrize(
    ("source", "blocked_by"),
    [("usage.md", None), ("cli.md", "citations")],
)
def test_team_checks_citations_against_writer_evidence(tmp_path, source, blocked_by):
    quote = "```bash\n$ poetry add pendulum\n```"

    def researcher(state):
        state["chunks"] = [{"source": "usage.md", "text": quote, "similarity": 0.8}]
        state["evidence_chunks"] = [
            *state["chunks"],
            {"source": "cli.md", "text": "Other CLI details"},
        ]
        state["evidence"] = [{"source": "usage.md", "quote": quote}]
        state["model_calls"] += 1
        return state

    def writer(state):
        state["support_answer"] = SupportAnswer(
            answerable=True,
            answer="Use poetry add requests",
            commands=["poetry add requests"],
            sources=[source],
            confidence="high",
        )
        state["model_calls"] += 1
        return state

    def reviewer(state):
        state["review"] = {"approved": True, "problems": []}
        state["model_calls"] += 1
        return state

    with patch.object(coordinator, "TRACE_DIR", tmp_path):
        result = coordinator.answer_with_team(
            "How do I add a dependency?",
            agent_functions={
                "router": _router,
                "researcher": researcher,
                "writer": writer,
                "reviewer": reviewer,
            },
        )
    assert result["blocked_by"] == blocked_by
    if blocked_by is None:
        assert result["commands"] == ["poetry add requests"]


@pytest.mark.parametrize(
    "question", ["What is the weather today?", "What is the Bitcoin price?"]
)
def test_out_of_scope_skips_other_agents(tmp_path, question):
    def out(state):
        state["route"] = {"category": "out_of_scope"}
        state["model_calls"] += 1
        return state

    with (
        patch.object(coordinator, "TRACE_DIR", tmp_path),
        patch.object(coordinator, "run_router", out),
        patch.object(
            coordinator, "run_researcher", side_effect=AssertionError("called")
        ),
    ):
        result = coordinator.answer_with_team(question)
    assert result["answer"] == REFUSAL
    assert result["blocked_by"] == "scope"


def test_model_call_budget(tmp_path):
    with (
        patch.object(coordinator, "TRACE_DIR", tmp_path),
        patch.object(coordinator, "run_router", _router),
        patch.object(
            coordinator, "run_researcher", side_effect=AssertionError("called")
        ),
        pytest.raises(RuntimeError, match="max_model_calls exceeded"),
    ):
        coordinator.answer_with_team("How?", max_model_calls=1)


def _response(content=None, tool=None, arguments="{}"):
    call = (
        SimpleNamespace(
            id="call-1", function=SimpleNamespace(name=tool, arguments=arguments)
        )
        if tool
        else None
    )
    message = SimpleNamespace(content=content, tool_calls=[call] if call else [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        model_dump=lambda **_: {"content": content, "tool": tool},
    )


def test_tool_loop_and_unknown_tool(tmp_path):
    with (
        patch(
            "hello_poetry.agent.chat_with_tools",
            side_effect=[_response(tool="missing"), _response("done")],
        ) as model,
        patch("hello_poetry.agent.TRACE_DIR", tmp_path),
    ):
        result = run_agent("How?", "prompt", [], {}, "fake", max_steps=2)
    assert result["final_answer"] == "done"
    assert result["steps"][0]["result"] == "Error: Unknown tool 'missing'."
    assert model.call_count == 2


def _bad_tool_call(name, code="tool_use_failed"):
    response = httpx.Response(
        400, request=httpx.Request("POST", "https://api.groq.com")
    )
    return BadRequestError(
        "Tool call validation failed",
        response=response,
        body={
            "error": {
                "code": code,
                "message": f"attempted to call tool '{name}' which was not in request.tools",
            }
        },
    )


def test_groq_retries_only_the_malformed_channel_tool_call():
    schemas = [{"type": "function", "function": {"name": "read_doc"}}]
    messages = [{"role": "user", "content": "How?"}]
    with patch.object(llm, "Groq") as client:
        create = client.return_value.chat.completions.create
        create.side_effect = [
            _bad_tool_call("read_doc<|channel|>commentary"),
            _response("done"),
        ]
        assert (
            llm.chat_with_tools(messages, "fake", schemas).choices[0].message.content
            == "done"
        )
    assert create.call_count == 2
    assert create.call_args_list[0] == create.call_args_list[1]
    assert create.call_args.kwargs["tools"] is schemas


@pytest.mark.parametrize(
    "errors, expected_calls",
    [
        ([_bad_tool_call("unknown")], 1),
        ([_bad_tool_call("read_doc<|channel|>commentary", "other")], 1),
        ([_bad_tool_call("read_doc<|channel|>commentary")] * 2, 2),
    ],
)
def test_groq_does_not_retry_other_or_repeated_bad_tool_calls(errors, expected_calls):
    with patch.object(llm, "Groq") as client:
        create = client.return_value.chat.completions.create
        create.side_effect = errors
        with pytest.raises(BadRequestError):
            llm.chat_with_tools([], "fake", [])
    assert create.call_count == expected_calls


def test_tool_loop_stops_at_limit(tmp_path):
    with (
        patch(
            "hello_poetry.agent.chat_with_tools",
            return_value=_response(tool="list_docs"),
        ),
        patch("hello_poetry.agent.TRACE_DIR", tmp_path),
    ):
        result = run_agent(
            "How?", "prompt", [], {"list_docs": list}, "fake", max_steps=1
        )
    assert result["stopped_by_limit"] is True
    assert result["final_answer"] is None


def test_researcher_compacts_overlapping_read_message_but_keeps_full_evidence(tmp_path):
    passage = "* `--no-plugins`: Disables plugins."
    document = "Introduction. " * 80 + passage + " Other CLI options." * 250
    search = [{"source": "cli.md", "text": passage, "similarity": 0.8}]
    with (
        patch(
            "hello_poetry.agent.chat_with_tools",
            side_effect=[
                _response(tool="search_docs", arguments='{"query":"plugins","k":3}'),
                _response(tool="read_doc", arguments='{"source":"cli.md"}'),
                _response("done"),
            ],
        ) as model,
        patch("hello_poetry.agent.TRACE_DIR", tmp_path),
    ):
        result = run_agent(
            "How do I disable plugins?",
            "prompt",
            [],
            {
                "search_docs": lambda query, k: search,
                "read_doc": lambda source: document,
            },
            "fake",
            max_steps=3,
            compact_read_doc=True,
        )
    tool_messages = [
        message["content"]
        for message in model.call_args_list[2].kwargs["messages"]
        if message["role"] == "tool"
    ]
    assert passage in tool_messages[0]
    assert passage not in tool_messages[1]
    assert len(tool_messages[1]) < len(json.dumps(document)) / 2
    assert json.loads(result["tool_results"][1]["result"]) == document


def test_researcher_keeps_full_read_when_search_has_no_matching_passage():
    full = json.dumps("Important text elsewhere in the document")
    earlier = [
        {
            "tool": "search_docs",
            "result": json.dumps([{"source": "other.md", "text": "Other"}]),
        }
    ]
    assert (
        _researcher_read_excerpt("read_doc", '{"source":"cli.md"}', full, earlier)
        == full
    )


def test_tool_failure_returns_error():
    def broken():
        raise RuntimeError("disk unavailable")

    assert "Tool execution failed" in execute_tool("broken", "{}", {"broken": broken})
    assert "Invalid tool arguments" in execute_tool(
        "broken", "not-json", {"broken": broken}
    )


def test_guarded_answer_blocks_unretrieved_citation():
    answer = SupportAnswer(
        answerable=True,
        answer="Run this",
        commands=[],
        sources=["fake.md"],
        confidence="low",
    )
    agent = {
        "final_answer": "draft",
        "model_calls": 1,
        "total_tokens": 0,
        "trace_path": "trace.json",
        "steps": [],
    }
    chunks = [{"source": "cli.md", "text": "poetry add flask", "similarity": 0.8}]
    with (
        patch(
            "hello_poetry.guardrails.check_scope",
            return_value=SimpleNamespace(in_scope=True),
        ),
        patch("hello_poetry.tools.search_docs", return_value=chunks),
        patch("hello_poetry.agent.run_agent", return_value=agent),
        patch("hello_poetry.schemas.get_structured_answer", return_value=(answer, 0)),
    ):
        result = guarded_answer("How?")
    assert result["answer"] == REFUSAL
    assert result["blocked_by"] == "citations"


def test_guarded_answer_uses_document_read_by_agent():
    answer = SupportAnswer(
        answerable=True,
        answer="Use poetry add flask",
        commands=["poetry add flask"],
        sources=["read.md"],
        confidence="high",
    )
    agent = {
        "final_answer": "draft",
        "model_calls": 1,
        "total_tokens": 0,
        "trace_path": "trace.json",
        "steps": [],
        "tool_results": [
            {
                "tool": "read_doc",
                "arguments": '{"source":"read.md"}',
                "result": json.dumps("Use poetry add flask"),
            }
        ],
    }
    chunks = [{"source": "related.md", "text": "related topic", "similarity": 0.8}]
    with (
        patch(
            "hello_poetry.guardrails.check_scope",
            return_value=SimpleNamespace(in_scope=True),
        ),
        patch("hello_poetry.tools.search_docs", return_value=chunks),
        patch("hello_poetry.agent.run_agent", return_value=agent),
        patch(
            "hello_poetry.schemas.get_structured_answer", return_value=(answer, 0)
        ) as writer,
    ):
        result = guarded_answer("How?")
    assert result["blocked_by"] is None
    assert "read.md" in writer.call_args.args[0][1]["content"]


def test_researcher_passes_read_evidence_to_writer():
    chunks = [{"source": "related.md", "text": "related topic", "similarity": 0.8}]
    research = {
        "final_answer": json.dumps(
            {"evidence": [{"source": "read.md", "quote": "poetry add flask"}]}
        ),
        "model_calls": 1,
        "total_tokens": 8,
        "trace_path": "trace.json",
        "tool_results": [
            {
                "tool": "read_doc",
                "arguments": '{"source":"read.md"}',
                "result": json.dumps("Use poetry add flask"),
            }
        ],
    }
    state = {
        "question": "How?",
        "model": "fake",
        "trace": [],
        "model_calls": 0,
        "total_tokens": 0,
        "remaining_calls": 3,
    }
    with (
        patch("hello_poetry.agents.search_docs", return_value=chunks),
        patch("hello_poetry.agents.run_agent", return_value=research) as model,
    ):
        result = run_researcher(state)
    assert result["evidence"][0]["source"] == "read.md"
    assert any(chunk["source"] == "read.md" for chunk in result["evidence_chunks"])
    assert model.call_args.kwargs["compact_read_doc"] is True


@pytest.mark.parametrize(
    ("question", "command"),
    [
        ("How do I create a new Poetry project?", "poetry new demo"),
        ("How do I add a Poetry dependency?", "poetry add requests"),
    ],
)
def test_researcher_prefers_direct_command_example(question, command):
    chunks = [
        {
            "source": "related.md",
            "text": "Poetry project background",
            "similarity": 0.8,
        },
        {"source": "overview.md", "text": "More Poetry context", "similarity": 0.7},
        {"source": "notes.md", "text": "General project notes", "similarity": 0.6},
        {"source": "usage.md", "text": f"```bash\n{command}\n```", "similarity": 0.5},
    ]
    assert select_research_chunks(question, chunks)[0] == chunks[3]
    research = {
        "final_answer": None,
        "model_calls": 1,
        "total_tokens": 0,
        "trace_path": "trace.json",
        "tool_results": [],
    }
    state = {
        "question": question,
        "model": "fake",
        "trace": [],
        "model_calls": 0,
        "total_tokens": 0,
        "remaining_calls": 3,
    }
    with (
        patch("hello_poetry.agents.search_docs", return_value=chunks) as search,
        patch("hello_poetry.agents.run_agent", return_value=research),
    ):
        result = run_researcher(state)
    search.assert_called_once_with(question, k=10)
    assert result["evidence"][0] == {"source": "usage.md", "quote": chunks[3]["text"]}


def test_researcher_does_not_promote_unrelated_or_below_threshold_passages():
    chunks = [
        {"source": "weather.md", "text": "Today's weather", "similarity": 0.8},
        {"source": "notes.md", "text": "General notes", "similarity": 0.7},
        {"source": "other.md", "text": "Other facts", "similarity": 0.6},
        {"source": "commands.md", "text": "poetry new demo", "similarity": 0.2},
    ]
    assert select_research_chunks("What is the weather today?", chunks) == chunks[:3]


def test_researcher_keeps_direct_dependency_example_in_writer_evidence():
    chunks = [
        {"source": "related.md", "text": "Dependency metadata", "similarity": 0.8},
        {"source": "other.md", "text": "Package background", "similarity": 0.7},
        {"source": "notes.md", "text": "General notes", "similarity": 0.6},
        {
            "source": "usage.md",
            "text": "```bash\n$ poetry add pendulum\n",
            "similarity": 0.5,
        },
    ]
    research = {
        "final_answer": json.dumps(
            {"evidence": [{"source": "related.md", "quote": "Dependency metadata"}]}
        ),
        "model_calls": 1,
        "total_tokens": 0,
        "trace_path": "trace.json",
        "tool_results": [],
    }
    state = {
        "question": "How do I add a dependency using Poetry?",
        "model": "fake",
        "trace": [],
        "model_calls": 0,
        "total_tokens": 0,
        "remaining_calls": 3,
    }
    with (
        patch("hello_poetry.agents.search_docs", return_value=chunks),
        patch("hello_poetry.agents.run_agent", return_value=research),
    ):
        result = run_researcher(state)
    assert result["chunks"][0]["source"] == "usage.md"
    assert {chunk["source"] for chunk in result["chunks"]} == {
        "related.md",
        "other.md",
        "notes.md",
        "usage.md",
    }
    assert result["evidence"][0] == {
        "source": "usage.md",
        "quote": chunks[3]["text"],
    }
    assert result["evidence"][1]["source"] == "related.md"


@pytest.mark.parametrize("as_list", [False, True])
def test_researcher_uses_fenced_verified_quote_without_dropping_tool_evidence(as_list):
    chunks = [{"source": "plugins.md", "text": "Plugin details", "similarity": 0.8}]
    quote = "* `--no-plugins`: Disables plugins."
    proposed = [{"source": "cli.md", "quote": quote}]
    content = proposed if as_list else {"evidence": proposed}
    research = {
        "final_answer": f"```json\n{json.dumps(content)}\n```",
        "model_calls": 2,
        "total_tokens": 50,
        "trace_path": "trace.json",
        "tool_results": [
            {
                "tool": "read_doc",
                "arguments": '{"source":"cli.md"}',
                "result": json.dumps(f"Global options\n{quote}\nMore CLI details"),
            }
        ],
    }
    state = {
        "question": "How do I disable plugins?",
        "model": "fake",
        "trace": [],
        "model_calls": 0,
        "total_tokens": 0,
        "remaining_calls": 3,
    }
    with (
        patch("hello_poetry.agents.search_docs", return_value=chunks),
        patch("hello_poetry.agents.run_agent", return_value=research),
    ):
        result = run_researcher(state)
    assert result["evidence"] == [{"source": "cli.md", "quote": quote}]
    assert [chunk["source"] for chunk in result["evidence_chunks"]] == [
        "plugins.md",
        "cli.md",
    ]


def test_evaluation_checkpoints_and_resumes(tmp_path):
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "id,type,question,expected_source,key_phrase,should_refuse\n"
        "Q01,normal,How?,cli.md,poetry add,False\n",
        encoding="utf-8",
    )
    result = {
        "answer": "Use poetry add",
        "sources": [],
        "refused": False,
        "model_calls": 1,
        "tokens": 5,
    }
    with (
        patch.object(evaluation, "CASES", cases),
        patch.object(evaluation, "RESULTS", tmp_path),
        patch.object(evaluation, "_retry", return_value=result) as api,
        patch.object(evaluation.time, "sleep"),
    ):
        path = evaluation.run("plain", limit=1)
        evaluation.run("plain", limit=1)
        with pytest.raises(ValueError, match="Only a saved error or refusal"):
            evaluation.run("plain", retry_id="Q01")
    assert api.call_count == 1
    assert "Use poetry add" in path.read_text(encoding="utf-8")


def test_evaluation_uses_project_env_without_network(tmp_path):
    (tmp_path / ".env").write_text("GROQ_API_KEY=offline-only-key\n", encoding="utf-8")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Use poetry add"))],
        usage=SimpleNamespace(total_tokens=7),
    )
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(llm, "PROJECT_ROOT", tmp_path),
        patch.object(llm, "Groq") as client,
    ):
        client.return_value.chat.completions.create.return_value = response
        result = evaluation._run_case({"question": "How?"}, "plain")
    client.assert_called_once_with(api_key="offline-only-key", max_retries=0)
    assert result["answer"] == "Use poetry add"
    assert result["tokens"] == 7


def test_evaluation_keeps_errors_distinct(tmp_path):
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "id,type,question,expected_source,key_phrase,should_refuse\n"
        "S01,special,Private key?,,,True\n",
        encoding="utf-8",
    )
    with (
        patch.object(evaluation, "CASES", cases),
        patch.object(evaluation, "RESULTS", tmp_path),
        patch.object(evaluation, "_retry", side_effect=RuntimeError("API failed")),
        patch.object(evaluation.time, "sleep"),
    ):
        path = evaluation.run("plain", limit=1)
    assert ",error," in path.read_text(encoding="utf-8")
    assert "API failed" in path.read_text(encoding="utf-8")


def test_evaluation_retries_refusal_only_when_explicit_and_keeps_original(tmp_path):
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "id,type,question,expected_source,key_phrase,should_refuse\n"
        "Q04,normal,How?,cli.md,--no-plugins,False\n",
        encoding="utf-8",
    )
    refusal = {"answer": REFUSAL, "refused": True, "model_calls": 4, "tokens": 30}
    corrected = {
        "answer": "Use --no-plugins",
        "refused": False,
        "model_calls": 4,
        "tokens": 32,
    }
    with (
        patch.object(evaluation, "CASES", cases),
        patch.object(evaluation, "RESULTS", tmp_path),
        patch.object(evaluation, "_retry", side_effect=[refusal, corrected]) as api,
        patch.object(evaluation.time, "sleep"),
    ):
        original = evaluation.run("single", limit=1)
        original_bytes = original.read_bytes()
        evaluation.run("single", limit=1)
        retry = evaluation.run("single", retry_id="Q04")
        evaluation.run("single", retry_id="Q04")
    assert api.call_count == 2
    assert original.read_bytes() == original_bytes
    assert next(csv.DictReader(original.open()))["status"] == "refused"
    assert next(csv.DictReader(retry.open()))["status"] == "completed"
    assert retry != original


def test_evaluation_retries_saved_error_without_repeating_success(tmp_path):
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "id,type,question,expected_source,key_phrase,should_refuse\n"
        "Q04,normal,How?,cli.md,--no-plugins,False\n",
        encoding="utf-8",
    )
    corrected = {
        "answer": "Use --no-plugins",
        "refused": False,
        "model_calls": 1,
        "tokens": 12,
    }
    with (
        patch.object(evaluation, "CASES", cases),
        patch.object(evaluation, "RESULTS", tmp_path),
        patch.object(
            evaluation, "_retry", side_effect=[RuntimeError("API failed"), corrected]
        ) as api,
        patch.object(evaluation.time, "sleep"),
    ):
        original = evaluation.run("team", limit=1)
        original_bytes = original.read_bytes()
        evaluation.run("team", limit=1)
        retry = evaluation.run("team", retry_id="Q04")
        with pytest.raises(ValueError, match="Retry output must differ"):
            evaluation.run("team", retry_id="Q04", retry_output=original)
    assert api.call_count == 2
    assert original.read_bytes() == original_bytes
    assert next(csv.DictReader(original.open()))["status"] == "error"
    assert next(csv.DictReader(retry.open()))["status"] == "completed"


def test_evaluation_stops_on_first_rate_limit():
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.groq.com")
    )
    error = RateLimitError("Rate limit reached", response=response, body={})
    with (
        patch.object(evaluation, "_run_case", side_effect=error) as api,
        patch.object(evaluation.time, "sleep") as pause,
        pytest.raises(RateLimitError),
    ):
        evaluation._retry({"question": "How?"}, "plain")
    api.assert_called_once()
    pause.assert_not_called()


def test_evaluation_stops_on_first_connection_error():
    error = APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
    with (
        patch.object(evaluation, "_run_case", side_effect=error) as api,
        patch.object(evaluation.time, "sleep") as pause,
        pytest.raises(APIConnectionError),
    ):
        evaluation._retry({"question": "How?"}, "plain")
    api.assert_called_once()
    pause.assert_not_called()


def test_single_results_export_after_guardrails_without_overwrite(tmp_path):
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "id,type,question,expected_source,key_phrase,should_refuse\n"
        "Q01,normal,How?,cli.md,poetry add,False\n",
        encoding="utf-8",
    )
    result = {
        "answer": "Use poetry add",
        "sources": ["cli.md"],
        "refused": False,
        "model_calls": 2,
        "tokens": 10,
    }
    with (
        patch.object(evaluation, "CASES", cases),
        patch.object(evaluation, "RESULTS", tmp_path),
        patch.object(evaluation, "_retry", return_value=result),
        patch.object(evaluation.time, "sleep"),
    ):
        path = evaluation.run("single", limit=1)
        assert (tmp_path / "after_guardrails.csv").read_bytes() == path.read_bytes()
        (tmp_path / "after_guardrails.csv").write_text("different", encoding="utf-8")
        with pytest.raises(FileExistsError, match="differs"):
            evaluation.run("single", limit=1)


def test_refusal_proxy_and_fixed_manual_sample():
    assert evaluation.explicit_doc_refusal("No relevant documentation was found.")
    assert not evaluation.explicit_doc_refusal("Use poetry add flask.")
    cases = {f"Q{i:02d}": {"should_refuse": "False"} for i in range(30)}
    cases.update({f"S{i:02d}": {"should_refuse": "True"} for i in range(10)})
    chosen = manual_grade_ids(cases)
    assert chosen == manual_grade_ids(cases)
    assert len(chosen) == 25 and len(set(chosen)) == 25


def test_support_team_page_loads_offline():
    page = Path(__file__).resolve().parents[1] / "pages" / "7_Support_Team.py"
    app = AppTest.from_file(page, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Poetry Support Team"
    app.button[0].click().run()
    assert "Please enter a question" in app.warning[0].value


def test_support_team_page_displays_mocked_answer(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-key")
    page = Path(__file__).resolve().parents[1] / "pages" / "7_Support_Team.py"
    fake_collection = Mock()
    fake_collection.count.return_value = 1
    fake_client = Mock()
    fake_client.get_collection.return_value = fake_collection
    result = {
        "answer": "Use poetry add flask",
        "commands": ["poetry add flask"],
        "sources": ["cli.md"],
        "confidence": "high",
        "blocked_by": None,
        "trace": [
            {
                "agent": "router",
                "input": "How?",
                "output": "how_to",
                "tokens": 3,
                "seconds": 0.1,
            }
        ],
        "model_calls": 4,
        "total_tokens": 20,
        "total_seconds": 1.0,
    }
    with (
        patch("chromadb.PersistentClient", return_value=fake_client),
        patch("hello_poetry.coordinator.answer_with_team", return_value=result),
    ):
        app = AppTest.from_file(page, default_timeout=30).run()
        app.text_area[0].set_value("How do I add flask?").run()
        app.button[0].click().run()
    assert not app.exception
    assert any("Use poetry add flask" in item.value for item in app.markdown)
    assert any("poetry add flask" in item.value for item in app.code)
