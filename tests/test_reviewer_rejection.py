
import coordinator


def test_reviewer_requests_revision_and_writer_corrects(monkeypatch):
    writer_calls = []
    reviewer_calls = []

    def fake_router_agent(*args, **kwargs):
        return "Category: commands"

    def fake_researcher_agent(*args, **kwargs):
        return "Poetry documentation: use poetry add to install packages."

    def fake_writer_agent(*args, **kwargs):
        writer_calls.append(kwargs)

        if len(writer_calls) == 1:
            return "Use poetry install to add a new package."

        return "Use poetry add <package-name> to add a package."

    def fake_reviewer_agent(*args, **kwargs):
        reviewer_calls.append(kwargs)

        if len(reviewer_calls) == 1:
            return {
                "review_decision": "revise",
                "review_feedback": (
                    "The answer is incorrect. "
                    "Use poetry add <package-name> instead."
                ),
                "answer": {
                    "answerable": True,
                    "answer": "Use poetry install to add a new package.",
                    "commands": ["poetry install"],
                    "sources": ["cli.md"],
                    "confidence": "low",
                },
            }

        return {
            "review_decision": "approve",
            "review_feedback": "",
            "answer": {
                "answerable": True,
                "answer": "Use poetry add <package-name> to add a package.",
                "commands": ["poetry add <package-name>"],
                "sources": ["cli.md"],
                "confidence": "high",
            },
        }

    monkeypatch.setattr(
        coordinator, "router_agent", fake_router_agent
    )
    monkeypatch.setattr(
        coordinator, "researcher_agent", fake_researcher_agent
    )
    monkeypatch.setattr(
        coordinator, "writer_agent", fake_writer_agent
    )
    monkeypatch.setattr(
        coordinator, "reviewer_agent", fake_reviewer_agent
    )

    result = coordinator.run_support_team(
        "How do I add a package using Poetry?",
        enable_reviewer=True,
    )

    # Writer should run twice: initial draft and revised answer.
    assert len(writer_calls) == 2

    # Reviewer should run twice: revise, then approve.
    assert len(reviewer_calls) == 2

    # Writer must receive the review feedback.
    assert "reviewer_feedback" in writer_calls[1]
    assert "incorrect" in writer_calls[1]["reviewer_feedback"]

    # Final answer must contain the corrected command.
    assert "poetry add" in result["answer"].answer
    assert "poetry install" not in result["answer"].answer

    # Final review decision should be approval.
    assert result["review_decision"] == "approve"
    assert result["review_count"] == 2
