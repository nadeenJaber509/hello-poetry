
from unittest.mock import patch

from agent import ModelCallBudgetExceeded
from coordinator import run_support_team


def test_coordinator_handles_budget_exhaustion():

    budget_error = ModelCallBudgetExceeded(
    "Maximum shared model-call budget reached."
)

    with patch(
        "coordinator.router_agent",
        side_effect=budget_error,
    ):
        try:
            result = run_support_team(
                "How do I install Poetry?"
            )
        except Exception as error:
            print("Test failed:", error)
            raise

    assert result["budget_exhausted"] is True
    assert result["answer"] is None
    assert result["model_calls"] == 0
    assert result["max_model_calls"] == 12

    print("PASS: Coordinator handles budget exhaustion.")


if __name__ == "__main__":
    test_coordinator_handles_budget_exhaustion()
