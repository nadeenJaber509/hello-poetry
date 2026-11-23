
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

load_dotenv()

from agents import (
    researcher_agent,
    reviewer_agent,
    router_agent,
    writer_agent,
)

MAX_MODEL_CALLS = 12
MAX_REVIEWS = 2


class SupportTeamState(TypedDict, total=False):
    question: str
    call_budget: dict
    route: str
    research: dict
    draft: str
    answer: dict
    review_decision: str
    reviewer_feedback: str
    review_count: int


@traceable(name="Router", run_type="chain")
def router_node(state: SupportTeamState) -> dict:
    route = router_agent(
        state["question"],
        state["call_budget"],
    )
    return {"route": route}


@traceable(name="Researcher", run_type="chain")
def researcher_node(state: SupportTeamState) -> dict:
    research = researcher_agent(
        state["question"],
        state["route"],
        state["call_budget"],
    )
    return {"research": research}


@traceable(name="Writer", run_type="chain")
def writer_node(state: SupportTeamState) -> dict:
    draft = writer_agent(
        state["question"],
        state["route"],
        state["research"],
        state["call_budget"],
        state.get("reviewer_feedback", ""),
    )
    return {"draft": draft}


@traceable(name="Reviewer", run_type="chain")
def reviewer_node(state: SupportTeamState) -> dict:
    result = reviewer_agent(
    state["question"],
    state["research"],
    state["draft"],
    state["call_budget"],
    include_review=True,
)

    review_count = state.get("review_count", 0) + 1

    return {
        "answer": result["answer"],
        "review_decision": result["review_decision"],
        "reviewer_feedback": result["review_feedback"],
        "review_count": review_count,
    }


def review_router(state: SupportTeamState) -> str:
    if state["review_decision"] == "approve":
        return "end"

    if state.get("review_count", 0) >= MAX_REVIEWS:
        return "end"

    return "revise"


def build_support_graph():
    workflow = StateGraph(SupportTeamState)

    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "router")
    workflow.add_edge("router", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "revise": "writer",
            "end": END,
        },
    )

    return workflow.compile()


support_graph = build_support_graph()


@traceable(
    name="LangGraph Support Team",
    run_type="chain",
    project_name="poetry-support-team",
)

def run_langgraph_support_team(question: str) -> dict:
    call_budget = {
        "max_calls": MAX_MODEL_CALLS,
        "used_calls": 0,
    }

    result = support_graph.invoke({
        "question": question,
        "call_budget": call_budget,
        "review_count": 0,
        "reviewer_feedback": "",
    })

    return {
        "answer": result["answer"],
        "route": result["route"],
        "research": result["research"],
        "draft": result["draft"],
        "review_decision": result["review_decision"],
        "review_count": result["review_count"],
        "reviewer_feedback": result["reviewer_feedback"],
        "model_calls": call_budget["used_calls"],
        "max_model_calls": MAX_MODEL_CALLS,
    }
