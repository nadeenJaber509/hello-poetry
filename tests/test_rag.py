
import rag

# Sample search results used by the tests.
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
    # Replace the search function with a fake function.
    def fake_hybrid_search(query, collection, model, top_k):
        assert query == "How do I create a Poetry project?"
        assert top_k == 3
        return SAMPLE_RESULTS

    # Replace the LLM function with a fake function.
    def fake_ask_llm(system_prompt, user_message, model, temperature):
        assert "poetry new my-package" in user_message
        assert temperature == 0
        return "Use poetry new my-package. [1]"

    monkeypatch.setattr(rag, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(rag, "ask_llm", fake_ask_llm)

    result = rag.answer_question(
        question="How do I create a Poetry project?",
        collection="fake_collection",
        model="fake_model",
        k=3,
    )

    assert result["answer"] == "Use poetry new my-package. [1]"
    assert len(result["sources"]) == 2
    assert result["sources"][0]["title"] == "Commands"
    assert result["sources"][0]["source"] == "cli.md"
    assert result["prompt"]["system"] == rag.SYSTEM_PROMPT


def test_answer_question_no_results(monkeypatch):
    # Simulate a search that returns no results.
    def fake_hybrid_search(query, collection, model, top_k):
        return []

    # This function should never be called.
    def fake_ask_llm(*args, **kwargs):
        raise AssertionError("LLM should not be called when no sources exist.")

    monkeypatch.setattr(rag, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(rag, "ask_llm", fake_ask_llm)

    result = rag.answer_question(
        question="What is the capital of France?",
        collection="fake_collection",
        model="fake_model",
        k=3,
    )

    assert result["sources"] == []
    assert result["prompt"] is None
    assert "could not find relevant information" in result["answer"].lower()
