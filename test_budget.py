
from unittest.mock import patch

from agent import ModelCallBudgetExceeded, _call_llm


def test_shared_budget_stops_at_limit():
    budget = {
        "max_calls": 12,
        "used_calls": 0,
    }

    fake_response = {
        "message": {"content": "Test response"},
        "usage": {"total_tokens": 10},
    }

    with patch(
        "agent.ask_llm_with_tools",
        return_value=fake_response,
    ) as mock_llm:

        for _ in range(12):
            _call_llm(
                messages=[],
                tools=None,
                model="test-model",
                call_budget=budget,
            )

        assert budget["used_calls"] == 12
        assert mock_llm.call_count == 12

        try:
            _call_llm(
                messages=[],
                tools=None,
                model="test-model",
                call_budget=budget,
            )
            raise AssertionError("Budget limit was not enforced")

        except ModelCallBudgetExceeded:
            pass

        assert budget["used_calls"] == 12
        assert mock_llm.call_count == 12

    print("PASS: Shared budget stops exactly at 12 calls.")


if __name__ == "__main__":
    test_shared_budget_stops_at_limit()


def test_retries_count_toward_budget():
    import httpx
    from groq import RateLimitError

    budget = {
        "max_calls": 3,
        "used_calls": 0,
    }

    response = httpx.Response(
        429,
        request=httpx.Request(
            "POST",
            "https://api.groq.com",
        ),
    )

    rate_limit_error = RateLimitError(
        "Rate limit reached",
        response=response,
        body=None,
    )

    with patch(
        "agent.ask_llm_with_tools",
        side_effect=rate_limit_error,
    ) as mock_llm:

        with patch("agent.time.sleep"):
            try:
                _call_llm(
                    messages=[],
                    tools=None,
                    model="test-model",
                    call_budget=budget,
                )
            except RateLimitError:
                pass
            else:
                raise AssertionError(
                    "Expected RateLimitError"
                )

        assert budget["used_calls"] == 3
        assert mock_llm.call_count == 3

    print("PASS: Retries count toward the shared budget.")

if __name__ == "__main__":
    test_shared_budget_stops_at_limit()
    test_retries_count_toward_budget()