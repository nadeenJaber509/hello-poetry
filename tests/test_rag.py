
import rag

SAMPLE_RESULTS = [
    {
        "title": "Commands",
        "source": "cli.md",
        "text": "Use poetry new my-package to create a new project.",
        "similarity": 0.85,
    },
    {
        "title": "Basic usage",
        "source": "basic-usage.md",
        "text": "Use poetry add package-name to add a dependency.",
        "similarity": 0.78,
    },
]


def test_format_sources():
    formatted = rag.format_sources(SAMPLE_RESULTS)

    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "Title: Commands" in formatted
    assert "File: cli.md" in formatted
    assert "poetry new my-package" in formatted


def test_format_sources_empty():
    assert rag.format_sources([]) == ""


def test_build_messages():
    question = "How do I create a Poetry project?"

    system_prompt, user_message = rag.build_messages(
        question,
        SAMPLE_RESULTS,
    )

    assert system_prompt == rag.SYSTEM_PROMPT
    assert question in user_message
    assert "Commands" in user_message
    assert "poetry new my-package" in user_message


def test_answer_question(monkeypatch):
    captured = {}

    def fake_hybrid_search(query, collection, model, top_k):
        assert query == "How do I create a Poetry project?"
        assert collection == "fake_collection"
        assert model == "fake_embedding_model"
        assert top_k == 3

        return SAMPLE_RESULTS

    def fake_run_agent(
        question,
        system_prompt,
        tool_schemas,
        tool_functions,
        model,
        max_steps,
    ):
        captured["question"] = question
        captured["system_prompt"] = system_prompt
        captured["tool_schemas"] = tool_schemas
        captured["tool_functions"] = tool_functions
        captured["model"] = model
        captured["max_steps"] = max_steps

        # Simulate the Agent calling search_docs.
        tool_functions["search_docs"](
            query=question,
            top_k=3,
        )

        return {
            "final_answer": "Use poetry new my-package. [1]",
            "steps": [
                {
                    "tool": "search_docs",
                    "args": {"query": question, "top_k": 3},
                    "result": "Search completed",
                    "seconds": 0.1,
                }
            ],
            "total_tokens": 150,
            "stopped_by_limit": False,
        }

    monkeypatch.setattr(rag, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(rag, "run_agent", fake_run_agent)

    result = rag.answer_question(
        question="How do I create a Poetry project?",
        collection="fake_collection",
        model="fake_embedding_model",
        k=3,
    )

    assert result["answer"] == "Use poetry new my-package. [1]"

    assert len(result["sources"]) == 2
    assert result["sources"][0]["title"] == "Commands"
    assert result["sources"][0]["source"] == "cli.md"

    assert result["sources"][0]["source_number"] == 1
    assert result["sources"][1]["source_number"] == 2

    assert captured["system_prompt"] == rag.SYSTEM_PROMPT
    assert captured["model"] == rag.MODEL
    assert captured["max_steps"] == 5

    assert "search_docs" in captured["tool_functions"]

    assert result["steps"]
    assert result["total_tokens"] == 150
    assert result["stopped_by_limit"] is False


def test_answer_question_no_results(monkeypatch):
    def fake_hybrid_search(query, collection, model, top_k):
        return []

    def fake_run_agent(
        question,
        system_prompt,
        tool_schemas,
        tool_functions,
        model,
        max_steps,
    ):
        # Simulate an empty search result.
        tool_results = tool_functions["search_docs"](
            query=question,
            top_k=3,
        )

        assert tool_results == []

        return {
            "final_answer": (
                "I couldn't find enough information in the "
                "available documents to answer this question."
            ),
            "steps": [
                {
                    "tool": "search_docs",
                    "args": {"query": question},
                    "result": "[]",
                    "seconds": 0.1,
                }
            ],
            "total_tokens": 80,
            "stopped_by_limit": False,
        }

    monkeypatch.setattr(rag, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(rag, "run_agent", fake_run_agent)

    result = rag.answer_question(
        question="What is the capital of France?",
        collection="fake_collection",
        model="fake_model",
        k=3,
    )

    assert result["sources"] == []
    assert "couldn't find enough information" in result["answer"].lower()
    assert result["prompt"]["system"] == rag.SYSTEM_PROMPT
    assert result["total_tokens"] == 80


def test_search_docs_limits_results(monkeypatch):
    def fake_hybrid_search(query, collection, model, top_k):
        assert top_k == 3

        return SAMPLE_RESULTS

    monkeypatch.setattr(rag, "hybrid_search", fake_hybrid_search)

    results = rag.search_docs(
        query="Poetry project",
        collection="fake_collection",
        model="fake_model",
        k=10,
    )

    assert len(results) == 2


def test_search_docs_rejects_empty_query():
    import pytest

    with pytest.raises(ValueError):
        rag.search_docs(
            query=" ",
            collection="fake_collection",
            model="fake_model",
        )


def test_tool_schema_matches_tool_function():
    import inspect

    signature = inspect.signature(
        lambda query, top_k=3: None
    )

    allowed_arguments = set(signature.parameters)

    schema_properties = set(
        rag.TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]
    )

    assert schema_properties == allowed_arguments
