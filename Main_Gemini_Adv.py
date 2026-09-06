import json
import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# Disable LangSmith tracing to prevent unnecessary network logging
os.environ["LANGCHAIN_TRACING_V2"] = "false"

load_dotenv()

# Verify API key availability
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Check your .env file.")


# ==========================================
# 1. CUSTOM GEMINI EMBEDDINGS (google-genai)
# ==========================================
class DirectGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model.replace("models/", "")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [item.values for item in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values


# ==========================================
# 2. PYDANTIC STRUCTURED OUTPUT SCHEMAS
# ==========================================
class RCAOutput(BaseModel):
    root_cause: str = Field(description="Detailed engineering analysis of the root cause.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 based on available data.")

class RecommendationOutput(BaseModel):
    corrective_action: str = Field(description="Immediate corrective action required.")
    preventive_action: str = Field(description="Long-term preventive maintenance step.")
    priority: str = Field(description="Priority level: High, Medium, or Low.")

class EvaluationOutput(BaseModel):
    faithfulness_score: float = Field(description="Score between 0.0 and 1.0 measuring how grounded the analysis is in context.")
    reasoning: str = Field(description="Brief justification for the assigned faithfulness score.")


# ==========================================
# 3. STATE DEFINITION
# ==========================================
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


# ==========================================
# 4. VECTOR DB & MODEL SETUP
# ==========================================
print("Setting up embedding model and vector database...")

embeddings = DirectGeminiEmbeddings(
    api_key=api_key,
    model="gemini-embedding-001"
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
        Document(page_content="High vibration and high bearing temperature indicate bearing failure."),
        Document(page_content="Low discharge pressure can be caused by worn bearings."),
        Document(page_content="Excessive noise and shaft misalignment can lead to premature bearing wear."),
        Document(page_content="High motor current may indicate pump overloading or internal mechanical damage."),
        Document(page_content="Low suction pressure may be caused by clogged filters or air leakage in the suction line."),
        Document(page_content="Frequent seal failures can result from improper installation or shaft misalignment."),
        Document(page_content="Increased vibration at specific frequencies often indicates imbalance in rotating parts."),
        Document(page_content="Overheating of motor windings can be caused by poor ventilation or electrical faults."),
        Document(page_content="Reduced flow rate may result from impeller wear or blockage in the system."),
        Document(page_content="Oil contamination in bearings can lead to increased friction and eventual failure."),
        Document(page_content="Cavitation noise and vibration occur due to low suction head or vapor formation in the fluid."),
        Document(page_content="Loose mounting bolts can cause excessive vibration and misalignment issues."),
    ]
    vector_db.add_documents(manual_docs)
    print("Data saved to disk at:", persist_directory)
else:
    print(f"Loaded existing database with {vector_db._collection.count()} items.")

# Active chat model setup
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
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
        {
            "incident_id": "INC-101",
            "issue": "High vibration detected during Q3 audit",
            "resolution": "Re-aligned shaft",
        },
        {
            "incident_id": "INC-204",
            "issue": "Bearing temp spike",
            "resolution": "Replaced bearing oil",
        },
    ]


# ==========================================
# 5. AGENT NODE DEFINITIONS
# ==========================================
def rag_agent(state: RCAState) -> Dict:
    print("[Agent] Running RAG Retrieval...")
    docs = vector_db.similarity_search(state["question"], k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return {"rag_context": context, "retry_count": state.get("retry_count", 0)}


def fallback_rag_agent(state: RCAState) -> Dict:
    print("[Agent] Running Expanded Fallback RAG Retrieval (k=6)...")
    docs = vector_db.similarity_search(state["question"], k=6)
    context = "\n".join([doc.page_content for doc in docs])
    return {
        "rag_context": context,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def sensor_agent(state: RCAState) -> Dict:
    print("[Agent] Fetching Live Sensor Data...")
    sensor_data = get_equipment_data(state["equipment_id"])
    return {"sensor_data": sensor_data}


def alarm_agent(state: RCAState) -> Dict:
    print("[Agent] Checking Active Alarms...")
    alarms = ["HIGH_VIBRATION", "HIGH_BEARING_TEMP", "LOW_DISCHARGE_PRESSURE"]
    return {"alarm_data": alarms}


def history_agent(state: RCAState) -> Dict:
    print("[Agent] Querying Historical Incidents...")
    incidents = get_historical_incidents(state["equipment_id"])
    return {"incident_history": incidents}


def rca_agent(state: RCAState) -> Dict:
    print("[Agent] Analyzing Root Cause with Structured Output...")
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

    return {
        "root_cause": result.root_cause,
        "confidence": result.confidence,
    }


def evaluator_agent(state: RCAState) -> Dict:
    print("[Agent] Evaluating Faithfulness & Hallucination...")
    prompt = f"""
    Evaluate the faithfulness of the proposed Root Cause Analysis against the Context.

    Context Provided:
    {state.get('rag_context', '')}
    Sensor Data: {json.dumps(state.get('sensor_data', {}))}
    Alarms: {state.get('alarm_data', [])}

    Proposed Root Cause:
    {state.get('root_cause', '')}

    Determine if the facts in the Root Cause Analysis are fully grounded in the provided Context.
    Score faithfulness from 0.0 (completely ungrounded / hallucinated) to 1.0 (fully grounded).
    """
    structured_evaluator = llm.with_structured_output(EvaluationOutput)
    eval_result: EvaluationOutput = structured_evaluator.invoke(prompt)

    is_hallucinated = eval_result.faithfulness_score < 0.70

    return {
        "faithfulness_score": eval_result.faithfulness_score,
        "is_hallucinated": is_hallucinated,
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
    print("[Agent] Compiling Final RCA Report...")
    rec = state.get("recommendation", {})
    report = f"""
==================================================
              ROOT CAUSE ANALYSIS REPORT
==================================================
Equipment ID     : {state.get('equipment_id', 'N/A')}
Issue Query      : {state.get('question', 'N/A')}

ROOT CAUSE:
{state.get('root_cause', 'N/A')}

METRICS & CONFIDENCE:
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

    # Trigger fallback RAG if confidence/faithfulness is low, up to 1 retry
    if (confidence < 0.80 or is_hallucinated) and retries < 1:
        print(f"[Router] Low confidence ({confidence:.2f}) or Hallucination flagged. Rerouting to Fallback RAG...")
        return "fallback_rag_agent"

    return "recommendation_agent"


# ==========================================
# 7. WORKFLOW GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(RCAState)

# Nodes
workflow.add_node("rag_agent", rag_agent)
workflow.add_node("fallback_rag_agent", fallback_rag_agent)
workflow.add_node("sensor_agent", sensor_agent)
workflow.add_node("alarm_agent", alarm_agent)
workflow.add_node("history_agent", history_agent)
workflow.add_node("rca_agent", rca_agent)
workflow.add_node("evaluator_agent", evaluator_agent)
workflow.add_node("recommendation_agent", recommendation_agent)
workflow.add_node("report_agent", report_agent)

# Start Fan-Out Edges
workflow.add_edge(START, "rag_agent")
workflow.add_edge(START, "sensor_agent")
workflow.add_edge(START, "alarm_agent")
workflow.add_edge(START, "history_agent")

# Parallel Processing Fan-In Edges to RCA Agent
workflow.add_edge("rag_agent", "rca_agent")
workflow.add_edge("sensor_agent", "rca_agent")
workflow.add_edge("alarm_agent", "rca_agent")
workflow.add_edge("history_agent", "rca_agent")

# RCA -> Evaluator Edge
workflow.add_edge("rca_agent", "evaluator_agent")

# Conditional Router Edge
workflow.add_conditional_edges(
    "evaluator_agent",
    route_after_evaluation,
    {
        "fallback_rag_agent": "fallback_rag_agent",
        "recommendation_agent": "recommendation_agent",
    },
)

# Retry Path back to RCA Agent
workflow.add_edge("fallback_rag_agent", "rca_agent")

# Final Sequence
workflow.add_edge("recommendation_agent", "report_agent")
workflow.add_edge("report_agent", END)

app = workflow.compile()

# ==========================================
# 8. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("\nStarting Enhanced RCA Graph Execution...\n")

    input_state = {
        "question": "Why is the pump experiencing high vibration and temperature spikes?",
        "equipment_id": "PUMP-K204",
    }

    final_state = app.invoke(input_state)
    print(final_state["report"])