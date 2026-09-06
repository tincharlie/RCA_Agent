from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field


# Pydantic Structured Outputs
class RCAOutput(BaseModel):
    root_cause: str = Field(description="Detailed engineering analysis of the root cause.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 based on available data.")


class RecommendationOutput(BaseModel):
    corrective_action: str = Field(description="Immediate corrective action required.")
    preventive_action: str = Field(description="Long-term preventive maintenance step.")
    priority: str = Field(description="Priority level: High, Medium, or Low.")


class EvaluationOutput(BaseModel):
    faithfulness_score: float = Field(
        description="Score between 0.0 and 1.0 measuring how grounded the analysis is in context."
    )
    reasoning: str = Field(description="Brief justification for the assigned faithfulness score.")


# LangGraph Central State
class RCAState(TypedDict, total=False):
    question: str
    equipment_id: str
    sensor_data: Dict[str, Any]
    alarm_data: List[str]
    incident_history: List[Dict[str, Any]]
    rag_context: str
    root_cause: str
    confidence: float
    recommendation: Dict[str, Any]
    faithfulness_score: float
    is_hallucinated: bool
    retry_count: int
    report: str