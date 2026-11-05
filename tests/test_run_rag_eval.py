"""Fast tests for the Task 9 evaluator; no network or model calls."""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.task9 import run_rag_eval as evaluation


def test_retrieval_hit_requires_same_chunk():
    chunks = [
        {"source": "correct.md", "text": "unrelated"},
        {"source": "other.md", "text": "Poetry ADD requests"},
    ]
    assert evaluation.retrieval_hit(chunks, "correct.md", "poetry add") == "false"
    assert evaluation.retrieval_hit(chunks, "other.md", "poetry add") == "true"
    assert evaluation.retrieval_hit(chunks, "", "") == ""


def test_load_questions(tmp_path: Path):
    path = tmp_path / "questions.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=evaluation.INPUT_COLUMNS)
        writer.writeheader()
        writer.writerow({"id": "1", "question": "How?", "expected_source": "a.md", "key_phrase": "x"})
    assert evaluation.load_questions(path)[0]["id"] == "1"


def test_plain_calls_llm_without_retrieval():
    with patch.object(evaluation, "ask_llm", return_value="ok") as llm, patch.object(
        evaluation, "answer_question"
    ) as rag:
        answer, sources = evaluation.get_answer("Question?", "plain", 3)
    assert (answer, sources) == ("ok", [])
    llm.assert_called_once_with(
        system_prompt=evaluation.PLAIN_SYSTEM_PROMPT,
        user_message="Question?",
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    rag.assert_not_called()


def test_rag_calls_existing_pipeline():
    expected = [{"source": "a.md", "text": "answer"}]
    with patch.object(evaluation, "answer_question", return_value={"answer": "ok", "sources": expected}) as rag:
        answer, sources = evaluation.get_answer("Question?", "rag", 3)
    assert (answer, sources) == ("ok", expected)
    rag.assert_called_once_with("Question?", "poetry_800_150", k=3)


def test_saved_answers_are_not_called_again(tmp_path: Path):
    question_path = tmp_path / "data" / "questions.csv"
    question_path.parent.mkdir()
    with question_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=evaluation.INPUT_COLUMNS)
        writer.writeheader()
        writer.writerow({"id": "1", "question": "Question?", "expected_source": "a.md", "key_phrase": "answer"})
    results_dir = tmp_path / "results"
    output = results_dir / "answers_plain_k3.csv"
    evaluation.save_rows(output, {"1": {
        "id": "1", "question": "Question?", "expected_source": "a.md", "key_phrase": "answer",
        "answer": "saved", "retrieved_sources": "", "retrieval_hit": "",
    }})
    with patch.object(evaluation, "QUESTIONS_PATH", question_path), patch.object(
        evaluation, "RESULTS_DIR", results_dir
    ), patch.object(evaluation, "get_answer_with_retry") as api:
        assert evaluation.run("plain", 3) == output
    api.assert_not_called()


def test_invalid_k():
    with pytest.raises(ValueError, match="k must"):
        evaluation.run("rag", k=0)
