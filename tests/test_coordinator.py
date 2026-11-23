
from unittest.mock import patch

from coordinator import run_support_team


def test_other_route_stops_remaining_agents():
    """Out-of-scope questions must stop after the Router."""

    with (
        patch(
            "coordinator.router_agent",
            return_value=(
                "Category: other\n"
                "Intent: Ask about cooking\n"
                "Research focus: None"
            ),
        ) as router,
        patch("coordinator.researcher_agent") as researcher,
        patch("coordinator.writer_agent") as writer,
        patch("coordinator.reviewer_agent") as reviewer,
    ):
        result = run_support_team("How do I cook pasta?")

    router.assert_called_once()
    researcher.assert_not_called()
    writer.assert_not_called()
    reviewer.assert_not_called()

    assert result["answer"] is not None
    assert result["answer"].answerable is False
    assert result["research"] is None
    assert result["draft"] is None
    assert result["model_calls"] == 0
    assert result["budget_exhausted"] is False

def _review_result(decision, answer, feedback=""):
    return {
        "review_decision": decision,
        "review_feedback": feedback,
        "answer": {
            "answerable": True,
            "answer": answer,
            "commands": ["poetry add requests"],
            "sources": ["basic-usage.md"],
            "confidence": "high",
        },
    }


def test_coordinator_stops_when_reviewer_approves(monkeypatch):
    import coordinator

    monkeypatch.setattr(
        coordinator,
        "router_agent",
        lambda *args, **kwargs: "Category: commands",
    )
    monkeypatch.setattr(
        coordinator,
        "researcher_agent",
        lambda **kwargs: {"collected": []},
    )
    monkeypatch.setattr(
        coordinator,
        "writer_agent",
        lambda **kwargs: "Initial draft",
    )

    reviewer_calls = []

    def fake_reviewer(**kwargs):
        reviewer_calls.append(kwargs)
        return _review_result("approve", "Approved answer")

    monkeypatch.setattr(coordinator, "reviewer_agent", fake_reviewer)

    result = coordinator.run_support_team("How do I add a package?")

    assert result["answer"].answer == "Approved answer"
    assert result["review_decision"] == "approve"
    assert result["review_count"] == 1
    assert len(reviewer_calls) == 1
    assert reviewer_calls[0]["include_review"] is True


def test_coordinator_revises_after_reviewer_feedback(monkeypatch):
    import coordinator

    monkeypatch.setattr(
        coordinator,
        "router_agent",
        lambda *args, **kwargs: "Category: commands",
    )
    monkeypatch.setattr(
        coordinator,
        "researcher_agent",
        lambda **kwargs: {"collected": []},
    )

    writer_drafts = []

    def fake_writer(**kwargs):
        writer_drafts.append(kwargs)
        return f"Draft {len(writer_drafts)}"

    monkeypatch.setattr(coordinator, "writer_agent", fake_writer)

    reviewer_calls = []

    def fake_reviewer(**kwargs):
        reviewer_calls.append(kwargs)

        if len(reviewer_calls) == 1:
            return _review_result(
                "revise",
                "First answer",
                "Correct the command explanation.",
            )

        return _review_result("approve", "Corrected answer")

    monkeypatch.setattr(coordinator, "reviewer_agent", fake_reviewer)

    result = coordinator.run_support_team("How do I add a package?")

    assert result["answer"].answer == "Corrected answer"
    assert result["review_decision"] == "approve"
    assert result["review_count"] == 2
    assert len(writer_drafts) == 2
    assert writer_drafts[1]["reviewer_feedback"] == (
        "Correct the command explanation."
    )
    assert reviewer_calls[1]["draft"] == "Draft 2"


def test_coordinator_stops_after_max_reviews(monkeypatch):
    import coordinator

    monkeypatch.setattr(
        coordinator,
        "router_agent",
        lambda *args, **kwargs: "Category: commands",
    )
    monkeypatch.setattr(
        coordinator,
        "researcher_agent",
        lambda **kwargs: {"collected": []},
    )
    monkeypatch.setattr(
        coordinator,
        "writer_agent",
        lambda **kwargs: "Draft",
    )

    reviewer_calls = []

    def fake_reviewer(**kwargs):
        reviewer_calls.append(kwargs)
        return _review_result(
            "revise",
            "Final reviewed answer",
            "Still needs revision.",
        )

    monkeypatch.setattr(coordinator, "reviewer_agent", fake_reviewer)

    result = coordinator.run_support_team("How do I add a package?")

    assert result["review_count"] == coordinator.MAX_REVIEWS
    assert len(reviewer_calls) == coordinator.MAX_REVIEWS
    assert result["review_decision"] == "revise"
    assert result["answer"].answer == "Final reviewed answer"
def test_coordinator_skips_reviewer_when_disabled(monkeypatch):
    from coordinator import run_support_team

    reviewer_calls = []

    monkeypatch.setattr(
        "coordinator.router_agent",
        lambda *args, **kwargs: "Category: usage",
    )

    monkeypatch.setattr(
        "coordinator.researcher_agent",
        lambda *args, **kwargs: {
            "collected": ["Poetry documentation"],
        },
    )

    monkeypatch.setattr(
        "coordinator.writer_agent",
        lambda *args, **kwargs: {
            "answerable": True,
            "answer": "Use poetry install.",
            "commands": ["poetry install"],
            "sources": ["basic-usage.md"],
            "confidence": "high",
        },
    )

    def fake_reviewer(*args, **kwargs):
        reviewer_calls.append(True)
        raise AssertionError("Reviewer should not be called")

    monkeypatch.setattr(
        "coordinator.reviewer_agent",
        fake_reviewer,
    )

    result = run_support_team(
        "How do I install dependencies?",
        enable_reviewer=False,
    )

    assert result["answer"].answer == "Use poetry install."
    assert result["review_decision"] == "skipped"
    assert result["review_count"] == 0
    assert reviewer_calls == []