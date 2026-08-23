from models.state import RCAState
# from services.openai_service import (get_llm)
from services.llm_service import (get_llm)

llm = get_llm()

def rca_agent(
        state: RCAState
):

    prompt = f"""

    You are a senior reliability engineer.

    Question:
    {state['question']}

    Manual Context:
    {state['rag_context']}

    Sensor Data:
    {state['sensor_data']}

    Alarm Data:
    {state['alarm_data']}

    Historical Incidents:
    {state['incident_history']}

    Provide:

    1 Root Cause

    2 Confidence Score

    3 Evidence

    """

    response = llm.invoke(
        prompt
    )

    state["root_cause"] = (
        response.content
    )

    state["confidence"] = 0.92

    return state
    