import json
import os
from typing import Any, Dict, List, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pymongo import MongoClient

os.environ["LANGCHAIN_TRACING_V2"] = "false"

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# mongodb_uri = os.getenv("MONGODB_URI")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
# if not mongodb_uri:
#     raise ValueError("MONGODB_URI environment variable is not set.")


# ==========================================
# 1. RETRIEVAL EVALUATOR
# ==========================================
class RetrievalEvaluator:
    @staticmethod
    def calculate_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        top_k_retrieved = retrieved_ids[:k]
        relevant_retrieved = set(top_k_retrieved).intersection(set(ground_truth_ids))
        return len(relevant_retrieved) / k if k > 0 else 0.0

    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0
        top_k_retrieved = retrieved_ids[:k]
        relevant_retrieved = set(top_k_retrieved).intersection(set(ground_truth_ids))
        return len(relevant_retrieved) / len(ground_truth_ids)

    @staticmethod
    def calculate_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        ground_truth_set = set(ground_truth_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_set:
                return 1.0 / rank
        return 0.0


# ==========================================
# 2. PYDANTIC SCHEMAS
# ==========================================
class RCAOutput(BaseModel):
    root_cause: str = Field(description="Detailed engineering analysis of the root cause.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")

class RecommendationOutput(BaseModel):
    corrective_action: str = Field(description="Immediate corrective action required.")
    preventive_action: str = Field(description="Long-term preventive maintenance step.")
    priority: str = Field(description="Priority level: High, Medium, or Low.")

class EvaluationOutput(BaseModel):
    faithfulness_score: float = Field(description="Score between 0.0 and 1.0 measuring groundedness.")
    reasoning: str = Field(description="Brief justification.")


# ==========================================
# 3. STATE DEFINITION
# ==========================================
class RCAState(TypedDict, total=False):
    question: str
    equipment_id: str
    ground_truth_docs: List[str]
    sensor_data: Dict[str, Any]
    alarm_data: List[str]
    incident_history: List[Dict[str, Any]]
    rag_context: str
    retrieved_doc_ids: List[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    root_cause: str
    confidence: float
    recommendation: Dict[str, Any]
    faithfulness_score: float
    is_hallucinated: bool
    retry_count: int
    report: str


# ==========================================
# 4. VECTOR DB SETUP (MONGODB ATLAS)
# ==========================================
print("Setting up MongoDB Atlas Vector Search...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

persist_directory = "./chroma_db"

vector_db = Chroma(
    collection_name="rca_manuals",
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

if vector_db._collection.count() == 0:
    print("Populating persistent database...")
    manual_docs = [
        Document(
            page_content=(
                "High vibration and high bearing temperature indicate bearing"
                " failure."
            )
        ),
        Document(
            page_content="Low discharge pressure can be caused by worn bearings."
        ),
        Document(
            page_content=(
                "Excessive noise and shaft misalignment can lead to premature"
                " bearing wear."
            )
        ),
        Document(
            page_content=(
                "High motor current may indicate pump overloading or internal"
                " mechanical damage."
            )
        ),
        Document(
            page_content=(
                "Low suction pressure may be caused by clogged filters or air"
                " leakage in the suction line."
            )
        ),
        Document(
            page_content=(
                "Frequent seal failures can result from improper installation"
                " or shaft misalignment."
            )
        ),
        Document(
            page_content=(
                "Increased vibration at specific frequencies often indicates"
                " imbalance in rotating parts."
            )
        ),
        Document(
            page_content=(
                "Overheating of motor windings can be caused by poor"
                " ventilation or electrical faults."
            )
        ),
        Document(
            page_content=(
                "Reduced flow rate may result from impeller wear or blockage in"
                " the system."
            )
        ),
        Document(
            page_content=(
                "Oil contamination in bearings can lead to increased friction"
                " and eventual failure."
            )
        ),
        Document(
            page_content=(
                "Cavitation noise and vibration occur due to low suction head or"
                " vapor formation in the fluid."
            )
        ),
        Document(
            page_content=(
                "Loose mounting bolts can cause excessive vibration and"
                " misalignment issues."
            )
        ),
    ]
    vector_db.add_documents(manual_docs)
    print("Data saved to disk at:", persist_directory)
else:
    print(
        f"Loaded existing database with {vector_db._collection.count()} items."
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=api_key,
    temperature=0,
)


def get_equipment_data(equipment_id: str):
    return {
        "vibration": 10.8,
        "bearing_temp": 112,
        "suction_pressure": 4.2,
        "discharge_pressure": 6.1,
        "flow": 50,
    }


def get_historical_incidents(equipment_id: str):
    return [
        {"incident_id": "INC-101", "issue": "High vibration detected during Q3 audit", "resolution": "Re-aligned shaft"},
        {"incident_id": "INC-204", "issue": "Bearing temp spike", "resolution": "Replaced bearing oil"},
    ]


# ==========================================
# 5. AGENT NODE DEFINITIONS
# ==========================================
def rag_agent(state: RCAState) -> Dict:
    print("[Agent] Running MongoDB Vector Retrieval & Metrics...")
    k = 3
    docs = vector_db.similarity_search(state["question"], k=k)
    
    retrieved_ids = [doc.metadata.get("doc_id", f"doc_{i}") for i, doc in enumerate(docs)]
    context = "\n".join([doc.page_content for doc in docs])
    ground_truth = state.get("ground_truth_docs", [])

    p_at_k = RetrievalEvaluator.calculate_precision_at_k(retrieved_ids, ground_truth, k=k)
    r_at_k = RetrievalEvaluator.calculate_recall_at_k(retrieved_ids, ground_truth, k=k)
    mrr = RetrievalEvaluator.calculate_reciprocal_rank(retrieved_ids, ground_truth)

    return {
        "rag_context": context,
        "retrieved_doc_ids": retrieved_ids,
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "mrr": mrr,
        "retry_count": state.get("retry_count", 0)
    }


def fallback_rag_agent(state: RCAState) -> Dict:
    print("[Agent] Running Fallback MongoDB Vector Retrieval (k=6)...")
    k = 6
    docs = vector_db.similarity_search(state["question"], k=k)
    
    retrieved_ids = [doc.metadata.get("doc_id", f"doc_{i}") for i, doc in enumerate(docs)]
    context = "\n".join([doc.page_content for doc in docs])
    ground_truth = state.get("ground_truth_docs", [])

    p_at_k = RetrievalEvaluator.calculate_precision_at_k(retrieved_ids, ground_truth, k=k)
    r_at_k = RetrievalEvaluator.calculate_recall_at_k(retrieved_ids, ground_truth, k=k)
    mrr = RetrievalEvaluator.calculate_reciprocal_rank(retrieved_ids, ground_truth)

    return {
        "rag_context": context,
        "retrieved_doc_ids": retrieved_ids,
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "mrr": mrr,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def sensor_agent(state: RCAState) -> Dict:
    return {"sensor_data": get_equipment_data(state["equipment_id"])}


def alarm_agent(state: RCAState) -> Dict:
    return {"alarm_data": ["HIGH_VIBRATION", "HIGH_BEARING_TEMP", "LOW_DISCHARGE_PRESSURE"]}


def history_agent(state: RCAState) -> Dict:
    return {"incident_history": get_historical_incidents(state["equipment_id"])}


def rca_agent(state: RCAState) -> Dict:
    print("[Agent] Analyzing Root Cause...")
    prompt = f"""
    You are a senior reliability engineer conducting Root Cause Analysis (RCA).

    User Question: {state.get('question', '')}
    Equipment ID: {state.get('equipment_id', '')}

    Manual & Documentation Context:
    {state.get('rag_context', 'N/A')}

    Live Sensor Data:
    {json.dumps(state.get('sensor_data', {}))}

    Triggered Alarms:
    {state.get('alarm_data', [])}

    Historical Incidents:
    {json.dumps(state.get('incident_history', []))}
    """
    structured_llm = llm.with_structured_output(RCAOutput)
    result: RCAOutput = structured_llm.invoke(prompt)

    return {"root_cause": result.root_cause, "confidence": result.confidence}


def evaluator_agent(state: RCAState) -> Dict:
    print("[Agent] Evaluating Faithfulness...")
    prompt = f"""
    Evaluate the faithfulness of the proposed Root Cause Analysis against the Context.

    Context Provided:
    {state.get('rag_context', '')}
    Sensor Data: {json.dumps(state.get('sensor_data', {}))}
    Alarms: {state.get('alarm_data', [])}

    Proposed Root Cause:
    {state.get('root_cause', '')}

    Determine if the facts in the Root Cause Analysis are fully grounded in the provided Context.
    Score faithfulness from 0.0 to 1.0.
    """
    structured_evaluator = llm.with_structured_output(EvaluationOutput)
    eval_result: EvaluationOutput = structured_evaluator.invoke(prompt)

    return {
        "faithfulness_score": eval_result.faithfulness_score,
        "is_hallucinated": eval_result.faithfulness_score < 0.70,
    }


def recommendation_agent(state: RCAState) -> Dict:
    print("[Agent] Generating Actionable Recommendations...")
    prompt = f"""
    Based on the following Root Cause Analysis:
    {state.get('root_cause', '')}

    Provide structured corrective and preventive actions with a priority level.
    """
    structured_rec = llm.with_structured_output(RecommendationOutput)
    result: RecommendationOutput = structured_rec.invoke(prompt)

    return {
        "recommendation": {
            "corrective_action": result.corrective_action,
            "preventive_action": result.preventive_action,
            "priority": result.priority,
        }
    }


def report_agent(state: RCAState) -> Dict:
    rec = state.get("recommendation", {})
    report = f"""
==================================================
              ROOT CAUSE ANALYSIS REPORT
==================================================
Equipment ID     : {state.get('equipment_id', 'N/A')}
Issue Query      : {state.get('question', 'N/A')}

RETRIEVAL METRICS (MongoDB Vector Search):
- Retrieved Doc IDs: {state.get('retrieved_doc_ids', [])}
- Precision@K      : {state.get('precision_at_k', 0.0):.4f}
- Recall@K         : {state.get('recall_at_k', 0.0):.4f}
- MRR              : {state.get('mrr', 0.0):.4f}

ROOT CAUSE:
{state.get('root_cause', 'N/A')}

ANALYSIS CONFIDENCE & FAITHFULNESS:
- Confidence Score : {state.get('confidence', 0.0):.2f}
- Faithfulness     : {state.get('faithfulness_score', 0.0):.2f}
- Hallucinated     : {state.get('is_hallucinated', False)}

RECOMMENDATIONS:
- Priority         : {rec.get('priority', 'N/A')}
- Corrective Action: {rec.get('corrective_action', 'N/A')}
- Preventive Action: {rec.get('preventive_action', 'N/A')}
==================================================
"""
    return {"report": report}


# ==========================================
# 6. ROUTING LOGIC
# ==========================================
def route_after_evaluation(state: RCAState) -> str:
    confidence = state.get("confidence", 0.0)
    is_hallucinated = state.get("is_hallucinated", False)
    retries = state.get("retry_count", 0)

    if (confidence < 0.80 or is_hallucinated) and retries < 1:
        print(f"[Router] Low confidence ({confidence:.2f}) or Hallucination flagged. Rerouting to Fallback...")
        return "fallback_rag_agent"

    return "recommendation_agent"


# ==========================================
# 7. WORKFLOW GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(RCAState)

workflow.add_node("rag_agent", rag_agent)
workflow.add_node("fallback_rag_agent", fallback_rag_agent)
workflow.add_node("sensor_agent", sensor_agent)
workflow.add_node("alarm_agent", alarm_agent)
workflow.add_node("history_agent", history_agent)
workflow.add_node("rca_agent", rca_agent)
workflow.add_node("evaluator_agent", evaluator_agent)
workflow.add_node("recommendation_agent", recommendation_agent)
workflow.add_node("report_agent", report_agent)

workflow.add_edge(START, "rag_agent")
workflow.add_edge(START, "sensor_agent")
workflow.add_edge(START, "alarm_agent")
workflow.add_edge(START, "history_agent")

workflow.add_edge("rag_agent", "rca_agent")
workflow.add_edge("sensor_agent", "rca_agent")
workflow.add_edge("alarm_agent", "rca_agent")
workflow.add_edge("history_agent", "rca_agent")

workflow.add_edge("rca_agent", "evaluator_agent")

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

app = workflow.compile()


# ==========================================
# 8. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\nStarting MongoDB-based RCA Graph Execution...\n")

    input_state = {
        "question": "Why is the pump experiencing high vibration and temperature spikes?",
        "equipment_id": "PUMP-K101",
        "ground_truth_docs": ["doc_1", "doc_3", "doc_10"],
    }

    final_state = app.invoke(input_state)
    print(final_state["report"])