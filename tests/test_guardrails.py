from agents import _sanitize_reviewer_answer


def test_sanitizer_removes_unsupported_toml_path():
    research = {"collected": [{"content": "Poetry manages project dependencies."}]}
    answer = "Add dependencies under [tool.poetry.dependencies]."

    result = _sanitize_reviewer_answer(answer, research)

    assert "[tool.poetry.dependencies]" not in result


def test_sanitizer_keeps_toml_path_supported_by_evidence():
    research = {
        "collected": [
            {"content": "The configuration uses [tool.poetry.dependencies]."}
        ]
    }
    answer = "Add dependencies under [tool.poetry.dependencies]."

    result = _sanitize_reviewer_answer(answer, research)

    assert "[tool.poetry.dependencies]" in result


def test_sanitizer_removes_unsupported_lockfile_update_claim():
    research = {"collected": [{"content": "Poetry manages dependencies."}]}
    answer = "The poetry.lock file is updated automatically."

    result = _sanitize_reviewer_answer(answer, research)

    assert "poetry.lock" not in result.lower()


def test_sanitizer_preserves_unrelated_answer():
    research = {"collected": [{"content": "Poetry is a Python dependency manager."}]}
    answer = "Run poetry install to install the project dependencies."

    result = _sanitize_reviewer_answer(answer, research)

    assert result == answer