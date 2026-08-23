from langgraph.graph import (StateGraph, END)
from models.state import RCAState
from agents.rag_agent import (rag_agent)
from agents.sensor_agent import (sensor_agent)
from agents.alarm_agent import (alarm_agent)
from agents.history_agent import (history_agent)
from agents.rca_agent import (rca_agent)
from agents.recommendation_agent import (recommendation_agent)
from agents.report_agent import (report_agent)

workflow = StateGraph(
    RCAState
)

workflow.add_node(
    "rag_agent",
    rag_agent
)

workflow.add_node(
    "sensor_agent",
    sensor_agent
)

workflow.add_node(
    "alarm_agent",
    alarm_agent
)

workflow.add_node(
    "history_agent",
    history_agent
)

workflow.add_node(
    "rca_agent",
    rca_agent
)

workflow.add_node(
    "recommendation_agent",
    recommendation_agent
)

workflow.add_node(
    "report_agent",
    report_agent
)

workflow.set_entry_point(
    "rag_agent"
)

workflow.add_edge(
    "rag_agent",
    "sensor_agent"
)


workflow.add_edge(
    "sensor_agent",
    "alarm_agent"
)


workflow.add_edge(
    "alarm_agent",
    "history_agent"
)


workflow.add_edge(
    "history_agent",
    "rca_agent"
)

workflow.add_edge(
    "rca_agent",
    "recommendation_agent"
)


workflow.add_edge(
    "recommendation_agent",
    "report_agent"
)

workflow.add_edge(
    "report_agent",
    END
)

graph = workflow.compile()