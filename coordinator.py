
import re

from groq import RateLimitError

from agent import ModelCallBudgetExceeded
from agents import (
    researcher_agent,
    reviewer_agent,
    router_agent,
    writer_agent,
)
from schemas import SupportAnswer

MAX_MODEL_CALLS = 12
MAX_REVIEWS = 2


def _is_out_of_scope(route: str) -> bool:
    """Check whether the Router classified the question as other."""

    if not isinstance(route, str):
        return False

    match = re.search(
        r"(?im)^\s*Category\s*:\s*([a-z_]+)\s*$",
        route,
    )

    if match:
        return match.group(1).strip().lower() == "other"

    return route.strip().lower() == "other"


def _build_unreviewed_answer(draft, research) -> SupportAnswer:
    """Convert the Writer's text into a valid SupportAnswer."""

    if isinstance(draft, SupportAnswer):
        return draft

    evidence_available = (
        isinstance(research, dict)
        and bool(research.get("collected"))
    )

    if isinstance(draft, dict):
        answer_data = draft.copy()
    else:
        answer_data = {"answer": str(draft)}

    answer_data.setdefault("answerable", evidence_available)
    answer_data.setdefault("commands", [])
    answer_data.setdefault(
    "sources",
    _extract_research_sources(research),
)
    answer_data.setdefault(
        "confidence",
        "medium" if evidence_available else "low",
    )

    return SupportAnswer.model_validate(answer_data)

def _extract_research_sources(research: dict) -> list[str]:
    """Extract documentation filenames from collected tool results."""
    import json

    sources = set()

    def collect(value):
        if isinstance(value, dict):
            source = value.get("source")
            if isinstance(source, str) and source:
                sources.add(source)

            for nested in value.values():
                collect(nested)

        elif isinstance(value, list):
            for item in value:
                collect(item)

    for item in research.get("collected", []):
        if not isinstance(item, dict):
            continue

        result = item.get("result", "")

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                continue

        collect(result)

    return sorted(sources)

def run_support_team(
    question: str,
    enable_reviewer: bool = True,
) -> dict:
    """Coordinate the four agents with review and a shared call budget."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    call_budget = {
        "max_calls": MAX_MODEL_CALLS,
        "used_calls": 0,
    }

    route = None
    research = None
    draft = None
    review_count = 0
    review_decision = None
    reviewer_feedback = ""

    try:
        # Step 1: Router
        route = router_agent(
            question,
            call_budget=call_budget,
        )

        if _is_out_of_scope(route):
            final_answer = SupportAnswer.model_validate(
                {
                    "answerable": False,
                    "answer": (
                        "This question is outside the scope of the "
                        "Poetry Support Team. I can only help with "
                        "Poetry-related questions."
                    ),
                    "commands": [],
                    "sources": [],
                    "confidence": "high",
                }
            )

            return {
                "answer": final_answer,
                "route": route,
                "research": None,
                "draft": None,
                "model_calls": call_budget["used_calls"],
                "max_model_calls": MAX_MODEL_CALLS,
                "budget_exhausted": False,
                "review_decision": None,
                "review_count": 0,
                "reviewer_feedback": "",
            }

        # Step 2: Researcher
        research = researcher_agent(
            question=question,
            route=route,
            call_budget=call_budget,
        )

        # Step 3: Initial Writer draft
        draft = writer_agent(
            question=question,
            route=route,
            research=research,
            call_budget=call_budget,
        )

        if not enable_reviewer:
            final_answer = _build_unreviewed_answer(
                draft,
                research,
            )
            review_decision = "skipped"

        else:
            # Steps 4-5: Reviewer and optional Writer revision
            for review_count in range(1, MAX_REVIEWS + 1):
                reviewed = reviewer_agent(
                    question=question,
                    research=research,
                    draft=draft,
                    call_budget=call_budget,
                    include_review=True,
                )

                review_decision = reviewed["review_decision"]
                reviewer_feedback = reviewed["review_feedback"]

                if review_decision == "approve":
                    break

                # Do not revise again after the final review.
                if review_count >= MAX_REVIEWS:
                    break

                # Send Reviewer feedback back to Writer.
                draft = writer_agent(
                    question=question,
                    route=route,
                    research=research,
                    call_budget=call_budget,
                    reviewer_feedback=reviewer_feedback,
                )

            # Step 6: Validate the reviewed structured answer.
            final_answer = SupportAnswer.model_validate(
                reviewed["answer"]
            )

        return {
            "answer": final_answer,
            "route": route,
            "research": research,
            "draft": draft,
            "model_calls": call_budget["used_calls"],
            "max_model_calls": MAX_MODEL_CALLS,
            "budget_exhausted": False,
            "review_decision": review_decision,
            "review_count": review_count,
            "reviewer_feedback": reviewer_feedback,
        }

    except RateLimitError as error:
        return {
            "answer": None,
            "route": route,
            "research": research,
            "draft": draft,
            "model_calls": call_budget["used_calls"],
            "max_model_calls": MAX_MODEL_CALLS,
            "budget_exhausted": False,
            "review_decision": review_decision,
            "review_count": review_count,
            "reviewer_feedback": reviewer_feedback,
            "error": (
                "Groq rate limit reached. "
                "Please check your API limits and try again later. "
                f"Details: {error}"
            ),
        }

    except ModelCallBudgetExceeded:
        return {
            "answer": None,
            "route": route,
            "research": research,
            "draft": draft,
            "model_calls": call_budget["used_calls"],
            "max_model_calls": MAX_MODEL_CALLS,
            "budget_exhausted": True,
            "review_decision": review_decision,
            "review_count": review_count,
            "reviewer_feedback": reviewer_feedback,
            "error": "Maximum model-call budget reached.",
        }
