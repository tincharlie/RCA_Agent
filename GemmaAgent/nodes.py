import json
from typing import Dict
from config import llm
from state import EvaluationOutput, RCAOutput, RCAState, RecommendationOutput
from vector_store import vector_db


# Mock External Services
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


# Graph Node Functions
def rag_agent(state: RCAState) -> Dict:
    docs = vector_db.similarity_search(state["question"], k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return {"rag_context": context, "retry_count": state.get("retry_count", 0)}


def fallback_rag_agent(state: RCAState) -> Dict:
    docs = vector_db.similarity_search(state["question"], k=6)
    context = "\n".join([doc.page_content for doc in docs])
    return {
        "rag_context": context,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def sensor_agent(state: RCAState) -> Dict:
    return {"sensor_data": get_equipment_data(state["equipment_id"])}


def alarm_agent(state: RCAState) -> Dict:
    return {"alarm_data": ["HIGH_VIBRATION", "HIGH_BEARING_TEMP", "LOW_DISCHARGE_PRESSURE"]}


def history_agent(state: RCAState) -> Dict:
    return {"incident_history": get_historical_incidents(state["equipment_id"])}


def rca_agent(state: RCAState) -> Dict:
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
    return {
        "faithfulness_score": eval_result.faithfulness_score,
        "is_hallucinated": eval_result.faithfulness_score < 0.70,
    }


def recommendation_agent(state: RCAState) -> Dict:
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