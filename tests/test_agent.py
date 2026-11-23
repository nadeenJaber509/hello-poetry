
import agent


def test_run_agent_returns_collected_tool_results(monkeypatch):
    """Ensure tool results are preserved in the agent output."""

    responses = [
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search_docs",
                            "arguments": '{"query": "Poetry dev dependency"}',
                        },
                    }
                ],
            },
            "usage": {"total_tokens": 10},
        },
        {
            "message": {
                "content": "Use poetry add --group dev.",
                "tool_calls": [],
            },
            "usage": {"total_tokens": 15},
        },
    ]

    def fake_call_llm(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    evidence = {
        "source": "cli.md",
        "text": "Use --group dev to add a development dependency.",
    }

    result = agent.run_agent(
        question="How do I add a development dependency?",
        system_prompt="Answer using documentation.",
        tool_schemas=[],
        tool_functions={
            "search_docs": lambda query: [evidence],
        },
        model="test-model",
        max_steps=2,
    )

    assert result["final_answer"] == "Use poetry add --group dev."
    assert len(result["collected"]) == 1
    assert result["collected"][0]["tool"] == "search_docs"
    assert result["collected"][0]["args"] == {
        "query": "Poetry dev dependency"
    }
    assert "cli.md" in result["collected"][0]["result"]
    assert "development dependency" in result["collected"][0]["result"]

def test_run_agent_stops_after_duplicate_search(monkeypatch):
    """Stop searching when a duplicate query is detected."""


    responses = [
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search_docs",
                            "arguments": (
                                '{"query": "Poetry lock resolved '
                                'dependency versions"}'
                            ),
                        },
                    }
                ],
            },
            "usage": {"total_tokens": 10},
        },
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "search_docs",
                            "arguments": (
                                '{"query": "Poetry lock resolved '
                                'dependency versions"}'
                            ),
                        },
                    }
                ],
            },
            "usage": {"total_tokens": 10},
        },
    ]

    def fake_call_llm(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    evidence = {
        "source": "basic-usage.md",
        "text": "The poetry.lock file records resolved dependency versions.",
    }

    def fake_force_final_answer(**kwargs):
        collected = kwargs["collected"]

        assert any(
            "basic-usage.md" in item["result"]
            for item in collected
        )

        return "The poetry.lock file records resolved versions.", 5

    monkeypatch.setattr(
        agent,
        "_force_final_answer",
        fake_force_final_answer,
    )

    result = agent.run_agent(
        question="What records resolved dependency versions?",
        system_prompt="Answer using documentation.",
        tool_schemas=[],
        tool_functions={
            "search_docs": lambda query: [evidence],
        },
        model="test-model",
        max_steps=3,
    )

    assert result["final_answer"] == (
        "The poetry.lock file records resolved versions."
    )
    assert len(responses) == 0
    assert result["stopped_by_limit"] is False
    assert any(
        "basic-usage.md" in item["result"]
        for item in result["collected"]
    )
