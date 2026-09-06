from langgraph.graph import END, START, StateGraph
from nodes import (
    alarm_agent,
    evaluator_agent,
    fallback_rag_agent,
    history_agent,
    rag_agent,
    rca_agent,
    recommendation_agent,
    report_agent,
    sensor_agent,
)
from state import RCAState


def route_after_evaluation(state: RCAState) -> str:
    confidence = state.get("confidence", 0.0)
    is_hallucinated = state.get("is_hallucinated", False)
    retries = state.get("retry_count", 0)

    if (confidence < 0.80 or is_hallucinated) and retries < 1:
        return "fallback_rag_agent"
    return "recommendation_agent"


def build_rca_graph():
    workflow = StateGraph(RCAState)

    # Add Nodes
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("fallback_rag_agent", fallback_rag_agent)
    workflow.add_node("sensor_agent", sensor_agent)
    workflow.add_node("alarm_agent", alarm_agent)
    workflow.add_node("history_agent", history_agent)
    workflow.add_node("rca_agent", rca_agent)
    workflow.add_node("evaluator_agent", evaluator_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)
    workflow.add_node("report_agent", report_agent)

    # Fan-Out Start Edges
    workflow.add_edge(START, "rag_agent")
    workflow.add_edge(START, "sensor_agent")
    workflow.add_edge(START, "alarm_agent")
    workflow.add_edge(START, "history_agent")

    # Fan-In Edges to RCA Agent
    workflow.add_edge("rag_agent", "rca_agent")
    workflow.add_edge("sensor_agent", "rca_agent")
    workflow.add_edge("alarm_agent", "rca_agent")
    workflow.add_edge("history_agent", "rca_agent")

    # RCA -> Evaluator
    workflow.add_edge("rca_agent", "evaluator_agent")

    # Conditional Routing
    workflow.add_conditional_edges(
        "evaluator_agent",
        route_after_evaluation,
        {
            "fallback_rag_agent": "fallback_rag_agent",
            "recommendation_agent": "recommendation_agent",
        },
    )

    workflow.add_edge("fallback_rag_agent", "rca_agent")
    workflow.add_edge("recommendation_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    return workflow.compile()


app_graph = build_rca_graph()