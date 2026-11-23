
import json

import agents


def test_writer_receives_collected_research(monkeypatch):
    """Ensure the writer receives the complete research evidence."""

    research = {
        "final_answer": "Research summary",
        "collected": [
            {
                "tool": "search_docs",
                "args": {"query": "development dependency"},
                "result": '{"source": "cli.md", "text": "Use --group dev."}',
            }
        ],
    }

    captured = {}

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {"final_answer": "Use --group dev."}

    monkeypatch.setattr(agents, "_run_agent", fake_run_agent)

    result = agents.writer_agent(
        question="How do I add a development dependency?",
        route="Documentation question",
        research=research,
    )

    assert result == "Use --group dev."
    assert "collected" in captured["question"]
    assert "cli.md" in captured["question"]
    assert "--group dev" in captured["question"]


def test_reviewer_receives_collected_research(monkeypatch):
    """Ensure the reviewer receives the complete research evidence."""

    research = {
        "final_answer": "Research summary",
        "collected": [
            {
                "tool": "search_docs",
                "args": {"query": "development dependency"},
                "result": '{"source": "cli.md", "text": "Use --group dev."}',
            }
        ],
    }

    captured = {}

    reviewer_response = {
        "answerable": True,
        "answer": "Use --group dev.",
        "commands": ["poetry add package --group dev"],
        "sources": ["cli.md"],
        "confidence": "high",
    }

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {
            "final_answer": json.dumps(reviewer_response),
        }

    monkeypatch.setattr(agents, "_run_agent", fake_run_agent)

    result = agents.reviewer_agent(
        question="How do I add a development dependency?",
        research=research,
        draft="Use --group dev.",
    )

    assert result == reviewer_response
    assert "collected" in captured["question"]
    assert "cli.md" in captured["question"]
    assert "--group dev" in captured["question"]
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "reviewer_response"
    assert response_format["json_schema"]["strict"] is True
