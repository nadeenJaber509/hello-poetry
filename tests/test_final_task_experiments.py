"""Offline checks for repeat runs and the optional Part 8 flow."""

import csv
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from groq import RateLimitError

from hello_poetry import team_graph
from hello_poetry.agent import evidence_from_tools
from hello_poetry.schemas import SupportAnswer
from scripts import (
    final_task_baseline,
    final_task_evaluate,
    final_task_part8,
    final_task_repeat,
)


def _cases(path: Path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "id",
                "type",
                "question",
                "should_refuse",
                "expected_source",
                "key_phrase",
            ),
        )
        writer.writeheader()
        for index in range(40):
            case_type = (
                "normal" if index < 30 else "special" if index < 37 else "injection"
            )
            writer.writerow(
                {
                    "id": f"Q{index:02d}",
                    "type": case_type,
                    "question": f"Question {index}?",
                    "should_refuse": str(case_type != "normal"),
                    "expected_source": (
                        final_task_repeat.NORMAL_SOURCES[index // 5]
                        if case_type == "normal"
                        else ""
                    ),
                    "key_phrase": "poetry add",
                }
            )


def test_fixed_sample_is_stratified_and_reproducible(tmp_path):
    cases = tmp_path / "cases.csv"
    _cases(cases)
    first = final_task_repeat.selected_cases(cases)
    assert first == final_task_repeat.selected_cases(cases)
    assert [case["type"] for case in first] == [
        *["normal"] * 6,
        *["special"] * 2,
        *["injection"] * 2,
    ]
    assert [case["expected_source"] for case in first[:6]] == list(
        final_task_repeat.NORMAL_SOURCES
    )


def test_sample_evaluation_resumes_and_stays_separate(tmp_path):
    cases = tmp_path / "cases.csv"
    _cases(cases)
    ids = [case["id"] for case in final_task_repeat.selected_cases(cases)]
    result = {
        "answer": "Use poetry add",
        "sources": [],
        "refused": False,
        "model_calls": 1,
        "tokens": 5,
    }
    with (
        patch.object(final_task_evaluate, "CASES", cases),
        patch.object(final_task_evaluate, "RESULTS", tmp_path),
        patch.object(final_task_evaluate, "_retry", return_value=result) as api,
        patch.object(final_task_evaluate.time, "sleep"),
    ):
        path = final_task_evaluate.run("plain", limit=1, sample_ten=True)
        final_task_evaluate.run("plain", limit=2, sample_ten=True)
        final_task_evaluate.run("plain", limit=2, sample_ten=True)
    assert path.name == "results_plain_sample10.csv"
    assert [row["id"] for row in csv.DictReader(path.open())] == ids[:2]
    assert api.call_count == 2
    assert not (tmp_path / "results_plain.csv").exists()


def test_sample_baseline_reuses_success_and_preserves_original(tmp_path):
    cases = tmp_path / "cases.csv"
    _cases(cases)
    selected = final_task_repeat.selected_cases(cases)
    original = tmp_path / "before_guardrails.csv"
    saved = {
        **selected[0],
        "final_answer": "saved answer",
        "model_calls": "1",
        "total_tokens": "10",
        "total_seconds": "1",
        "stopped_by_limit": "False",
        "steps": "[]",
        "trace_path": "trace.json",
        "status": "completed",
        "error": "",
    }
    final_task_baseline.save_results([saved], original)
    original_bytes = original.read_bytes()

    def fake_evaluate(case, _model, _steps):
        return {**saved, **case, "final_answer": "new answer"}

    with (
        patch.object(final_task_baseline, "CASES_FILE", cases),
        patch.object(final_task_baseline, "OUTPUT_FILE", original),
        patch.object(
            final_task_baseline, "evaluate_case", side_effect=fake_evaluate
        ) as api,
        patch.object(sys, "argv", ["baseline", "--sample-ten", "--limit", "1"]),
    ):
        final_task_baseline.main()
    sample = tmp_path / "before_guardrails_sample10.csv"
    assert [row["id"] for row in csv.DictReader(sample.open())] == [
        selected[0]["id"],
        selected[1]["id"],
    ]
    assert api.call_count == 1
    assert original.read_bytes() == original_bytes


def test_repeat_checkpoints_two_runs_and_compares(tmp_path):
    cases = tmp_path / "cases.csv"
    _cases(cases)
    calls = []

    def fake_retry(case, system):
        calls.append((case["id"], system))
        return {
            "answer": f"answer {len(calls)}",
            "refused": False,
            "sources": [],
            "model_calls": 1,
            "tokens": 5,
        }

    with (
        patch.object(final_task_repeat, "_retry", fake_retry),
        patch.object(final_task_repeat.time, "sleep"),
    ):
        path = final_task_repeat.run(
            "plain", 1, limit=2, cases_path=cases, results_dir=tmp_path
        )
        final_task_repeat.run(
            "plain", 1, limit=2, cases_path=cases, results_dir=tmp_path
        )
        final_task_repeat.run(
            "plain", 2, limit=2, cases_path=cases, results_dir=tmp_path
        )
    assert len(calls) == 4
    assert final_task_repeat.compare(path) == {
        "paired": 2,
        "usable": 2,
        "different_answers": 2,
        "different_ids": [
            case["id"]
            for case in sorted(
                final_task_repeat.selected_cases(cases)[:2], key=lambda case: case["id"]
            )
        ],
    }


def test_repeat_requires_first_run_and_preserves_settings(tmp_path):
    cases = tmp_path / "cases.csv"
    _cases(cases)
    with pytest.raises(ValueError, match="repeat 1"):
        final_task_repeat.run(
            "plain", 2, limit=1, cases_path=cases, results_dir=tmp_path
        )


def test_tool_evidence_includes_read_document():
    original = [{"source": "a.md", "text": "intro", "similarity": 0.7}]
    tool_results = [
        {
            "tool": "search_docs",
            "arguments": "{}",
            "result": json.dumps([{"source": "c.md", "text": "less useful"}]),
        },
        {
            "tool": "read_doc",
            "arguments": '{"source":"b.md"}',
            "result": json.dumps("Use poetry add flask"),
        },
        {
            "tool": "read_doc",
            "arguments": '{"source":"missing.md"}',
            "result": "Error: missing",
        },
    ]
    found = evidence_from_tools(original, tool_results, max_chunks=2)
    assert [chunk["source"] for chunk in found] == ["a.md", "b.md"]
    assert found[1]["text"] == "Use poetry add flask"


def test_tool_evidence_deduplicates_initial_passages():
    same = {"source": "cli.md", "text": "Use --no-plugins", "similarity": 0.8}
    found = evidence_from_tools([same, dict(same)], [], max_chunks=6)
    assert found == [same]


class _FakeStateGraph:
    def __init__(self, _schema):
        self.nodes = {}
        self.edges = {}
        self.routes = {}

    def add_node(self, name, function):
        self.nodes[name] = function
        return self

    def add_edge(self, source, target):
        self.edges[source] = target
        return self

    def add_conditional_edges(self, source, route):
        self.routes[source] = route
        return self

    def compile(self):
        return self

    def invoke(self, state, _config):
        current = self.edges["START"]
        while current != "END":
            state = self.nodes[current](state)
            current = (
                self.routes[current](state)
                if current in self.routes
                else self.edges[current]
            )
        return state


def test_graph_uses_conditional_review_loop_and_read_evidence(tmp_path):
    graph_package = ModuleType("langgraph")
    graph_module = ModuleType("langgraph.graph")
    graph_module.START, graph_module.END = "START", "END"
    graph_module.StateGraph = _FakeStateGraph
    graph_package.graph = graph_module
    writer_feedback = []
    reviewed_answers = []

    def advance(state, name):
        state["model_calls"] += 1
        state["trace"].append({"agent": name})
        return state

    def router(state):
        state["route"] = {"category": "how_to"}
        return advance(state, "router")

    def researcher(state):
        state["chunks"] = [{"source": "a.md", "text": "related", "similarity": 0.8}]
        state["evidence_chunks"] = [
            *state["chunks"],
            {"source": "b.md", "text": "poetry add flask"},
        ]
        return advance(state, "researcher")

    def writer(state):
        writer_feedback.append(state.get("review_problems", []))
        state["support_answer"] = SupportAnswer(
            answerable=True,
            answer=(
                "Use poetry add flask"
                if state.get("review_problems")
                else "Draft needs review"
            ),
            commands=["poetry add flask"] if state.get("review_problems") else [],
            sources=["b.md"],
            confidence="high",
        )
        return advance(state, "writer")

    def reviewer(state):
        reviewed_answers.append(state["support_answer"].answer)
        state["review"] = {
            "approved": state["review_rounds"] == 1,
            "problems": ["Check the command"],
        }
        return advance(state, "reviewer")

    with (
        patch.dict(
            sys.modules, {"langgraph": graph_package, "langgraph.graph": graph_module}
        ),
        patch.object(team_graph, "TRACE_DIR", tmp_path),
    ):
        result = team_graph.answer_with_graph(
            "How?",
            agent_functions={
                "router": router,
                "researcher": researcher,
                "writer": writer,
                "reviewer": reviewer,
            },
        )
    assert result["answer"] == "Use poetry add flask"
    assert result["answerable"] is True
    assert result["review_rounds"] == 2
    assert result["model_calls"] == 6
    assert [step["agent"] for step in result["trace"]] == [
        "router",
        "researcher",
        "writer",
        "reviewer",
        "writer",
        "reviewer",
    ]
    assert writer_feedback == [[], ["Check the command"]]
    assert reviewed_answers == ["Draft needs review", "Use poetry add flask"]


@pytest.mark.parametrize(
    "max_review_rounds, max_model_calls, expected_error",
    [(1, 12, None), (2, 4, "max_model_calls exceeded")],
)
def test_graph_stops_after_review_limit_or_model_call_limit(
    tmp_path, max_review_rounds, max_model_calls, expected_error
):
    graph_package = ModuleType("langgraph")
    graph_module = ModuleType("langgraph.graph")
    graph_module.START, graph_module.END = "START", "END"
    graph_module.StateGraph = _FakeStateGraph
    graph_package.graph = graph_module
    visited = []

    def advance(state, name):
        visited.append(name)
        state["model_calls"] += 1
        state["trace"].append({"agent": name})
        return state

    def router(state):
        state["route"] = {"category": "how_to"}
        return advance(state, "router")

    def researcher(state):
        state["chunks"] = [
            {"source": "cli.md", "text": "poetry add flask", "similarity": 0.8}
        ]
        return advance(state, "researcher")

    def writer(state):
        state["support_answer"] = SupportAnswer(
            answerable=True,
            answer="Draft",
            commands=[],
            sources=["cli.md"],
            confidence="low",
        )
        return advance(state, "writer")

    def reviewer(state):
        state["review"] = {"approved": False, "problems": ["Revise"]}
        return advance(state, "reviewer")

    functions = {
        "router": router,
        "researcher": researcher,
        "writer": writer,
        "reviewer": reviewer,
    }
    with (
        patch.dict(
            sys.modules, {"langgraph": graph_package, "langgraph.graph": graph_module}
        ),
        patch.object(team_graph, "TRACE_DIR", tmp_path),
    ):
        if expected_error:
            with pytest.raises(RuntimeError, match=expected_error):
                team_graph.answer_with_graph(
                    "How?",
                    max_review_rounds=max_review_rounds,
                    max_model_calls=max_model_calls,
                    agent_functions=functions,
                )
        else:
            result = team_graph.answer_with_graph(
                "How?",
                max_review_rounds=max_review_rounds,
                max_model_calls=max_model_calls,
                agent_functions=functions,
            )
            assert result["blocked_by"] == "reviewer"
            assert result["answerable"] is False
            assert result["review_rounds"] == 1
            assert result["model_calls"] == 4
    assert visited == ["router", "researcher", "writer", "reviewer"]


def test_graph_scope_route_skips_researcher(tmp_path):
    graph_package = ModuleType("langgraph")
    graph_module = ModuleType("langgraph.graph")
    graph_module.START, graph_module.END = "START", "END"
    graph_module.StateGraph = _FakeStateGraph
    graph_package.graph = graph_module

    def router(state):
        state["route"] = {"category": "out_of_scope"}
        state["model_calls"] += 1
        return state

    def unexpected(_state):
        raise AssertionError("Researcher should not run")

    with (
        patch.dict(
            sys.modules, {"langgraph": graph_package, "langgraph.graph": graph_module}
        ),
        patch.object(team_graph, "TRACE_DIR", tmp_path),
    ):
        result = team_graph.answer_with_graph(
            "Weather?",
            agent_functions={
                "router": router,
                "researcher": unexpected,
                "writer": unexpected,
                "reviewer": unexpected,
            },
        )
    assert result["blocked_by"] == "scope"
    assert result["model_calls"] == 1


def test_graph_comparison_resumes_saved_cases(tmp_path):
    case = {"id": "Q01", "question": "How?"}
    answer = {"answer": "Use poetry add", "model_calls": 2}
    with (
        patch.object(final_task_part8, "build_graph"),
        patch.object(final_task_part8, "selected_cases", return_value=[case]),
        patch.object(final_task_part8, "answer_with_team", return_value=answer) as team,
        patch.object(
            final_task_part8, "answer_with_graph", return_value=answer
        ) as graph,
        patch.object(final_task_part8.time, "sleep"),
    ):
        path = tmp_path / "comparison.json"
        rows = final_task_part8.compare_graph(1, output=path)
        final_task_part8.compare_graph(1, output=path)
    assert team.call_count == graph.call_count == 1
    assert final_task_part8.graph_summary(rows)["same_answer"] == 1


def test_graph_comparison_can_run_only_named_sample_cases(tmp_path):
    cases = [
        {"id": "Q23", "question": "Normal?"},
        {"id": "Q10", "question": "Special?"},
        {"id": "I03", "question": "Injection?"},
    ]
    output = tmp_path / "comparison_sample10.json"
    with (
        patch.object(final_task_part8, "build_graph"),
        patch.object(final_task_part8, "selected_cases", return_value=cases),
        patch.object(
            final_task_part8, "answer_with_team", return_value={"answer": "x"}
        ) as team,
        patch.object(
            final_task_part8, "answer_with_graph", return_value={"answer": "x"}
        ) as graph,
        patch.object(final_task_part8.time, "sleep"),
    ):
        rows = final_task_part8.compare_graph(output=output, case_ids=["Q10", "I03"])
        final_task_part8.compare_graph(output=output, case_ids=["Q10", "I03"])
    assert list(rows) == [
        "Q10:coordinator",
        "Q10:graph",
        "I03:coordinator",
        "I03:graph",
    ]
    assert team.call_count == graph.call_count == 2


def test_retry_failed_graph_runs_only_graph_and_preserves_comparison(tmp_path):
    comparison = tmp_path / "comparison.json"
    original = {
        "Q05:coordinator": {"question": "How?", "status": "completed", "result": {}},
        "Q05:graph": {"question": "How?", "status": "error", "error": "old failure"},
    }
    comparison.write_text(json.dumps(original), encoding="utf-8")
    output = tmp_path / "retry.json"
    with (
        patch.object(final_task_part8, "answer_with_team") as team,
        patch.object(
            final_task_part8, "answer_with_graph", return_value={"answer": "done"}
        ) as graph,
    ):
        saved = final_task_part8.retry_failed_graph("Q05", comparison, output)
        assert final_task_part8.retry_failed_graph("Q05", comparison, output) == saved
    assert saved["status"] == "completed"
    assert graph.call_count == 1
    team.assert_not_called()
    assert json.loads(comparison.read_text(encoding="utf-8")) == original


def test_retry_failed_graph_requires_failed_case_and_separate_output(tmp_path):
    comparison = tmp_path / "comparison.json"
    rows = {
        "Q05:coordinator": {"question": "How?", "status": "completed"},
        "Q05:graph": {"question": "How?", "status": "error"},
    }
    comparison.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        final_task_part8.retry_failed_graph("Q05", comparison, comparison)
    with pytest.raises(ValueError, match="No completed coordinator case"):
        final_task_part8.retry_failed_graph("Q06", comparison, tmp_path / "retry.json")


def test_trace_ten_resumes_without_retrying_saved_rows(tmp_path):
    cases = [
        {"id": "Q04", "question": "Completed?"},
        {"id": "Q27", "question": "Failed?"},
    ]
    output = tmp_path / "traces.json"
    saved = {
        "Q04": {"question": "Completed?", "status": "completed", "result": {}},
        "Q27": {"question": "Failed?", "status": "error", "error": "old 429"},
    }
    output.write_text(json.dumps(saved), encoding="utf-8")
    with (
        patch.object(final_task_part8, "selected_cases", return_value=cases),
        patch.object(final_task_part8, "_langsmith") as langsmith,
        patch.object(final_task_part8, "traced_team") as team,
    ):
        assert final_task_part8.trace_ten(2, output=output) == saved
    assert json.loads(output.read_text(encoding="utf-8")) == saved
    langsmith.assert_not_called()
    team.assert_not_called()


def test_trace_ten_explicit_retry_preserves_error_history(tmp_path):
    case = {"id": "Q27", "question": "Retry?"}
    output = tmp_path / "traces.json"
    original = {"question": "Retry?", "status": "error", "error": "first failure"}
    output.write_text(json.dumps({"Q27": original}), encoding="utf-8")
    with (
        patch.object(final_task_part8, "selected_cases", return_value=[case]),
        patch.object(final_task_part8, "_langsmith"),
        patch.object(
            final_task_part8,
            "_retry",
            side_effect=[RuntimeError("second failure"), {"answer": "done"}],
        ) as retry,
        patch.object(final_task_part8.time, "sleep"),
    ):
        failed = final_task_part8.trace_ten(1, output=output, retry_errors=True)
        assert failed["Q27"]["history"] == [original]
        completed = final_task_part8.trace_ten(1, output=output, retry_errors=True)
        final_task_part8.trace_ten(1, output=output, retry_errors=True)
    assert retry.call_count == 2
    assert completed["Q27"]["status"] == "completed"
    assert [item["error"] for item in completed["Q27"]["history"]] == [
        "first failure",
        "RuntimeError: second failure",
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == completed


def test_trace_ten_stops_on_first_429_without_retrying_next_case(tmp_path):
    cases = [
        {"id": "Q27", "question": "First?"},
        {"id": "Q30", "question": "Second?"},
    ]
    output = tmp_path / "traces.json"
    saved = {
        case["id"]: {
            "question": case["question"],
            "status": "error",
            "error": "previous 429",
        }
        for case in cases
    }
    output.write_text(json.dumps(saved), encoding="utf-8")
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.groq.com")
    )
    error = RateLimitError("TPM limit 8000", response=response, body={})
    with (
        patch.object(final_task_part8, "selected_cases", return_value=cases),
        patch.object(final_task_part8, "_langsmith"),
        patch.object(final_task_part8, "traced_team", side_effect=error) as team,
        patch.object(final_task_part8.time, "sleep") as sleep,
    ):
        rows = final_task_part8.trace_ten(2, output=output, retry_errors=True)
    team.assert_called_once_with("First?")
    sleep.assert_not_called()
    assert rows["Q27"]["status"] == "error"
    assert rows["Q27"]["history"] == [saved["Q27"]]
    assert rows["Q30"] == saved["Q30"]
    assert json.loads(output.read_text(encoding="utf-8")) == rows


def test_langsmith_upload_is_prepared_without_network(tmp_path):
    cases = tmp_path / "cases.csv"
    _cases(cases)
    received = []

    class FakeClient:
        def has_dataset(self, *, dataset_name):
            return False

        def create_dataset(self, *, dataset_name):
            return SimpleNamespace(id="dataset-id")

        def create_examples(self, *, dataset_id, examples):
            received.extend(examples)

    with patch.object(
        final_task_part8, "_langsmith", return_value=(FakeClient, None, None)
    ):
        assert final_task_part8.upload_dataset(cases_path=cases) == "dataset-id"
    assert len(received) == 40
    assert set(received[0]) == {"inputs", "outputs"}


def test_langsmith_tracing_wraps_existing_agents():
    names = []

    def traceable(*, name):
        def decorate(function):
            names.append(name)
            return function

        return decorate

    with (
        patch.object(
            final_task_part8, "_langsmith", return_value=(None, None, traceable)
        ),
        patch.object(
            final_task_part8, "answer_with_team", return_value={"answer": "ok"}
        ) as coordinator,
    ):
        assert final_task_part8.traced_team("How?") == {"answer": "ok"}
    assert names == [
        "router",
        "researcher",
        "writer",
        "reviewer",
        "poetry_support_team",
    ]
    assert set(coordinator.call_args.kwargs["agent_functions"]) == set(
        final_task_part8.AGENTS
    )
