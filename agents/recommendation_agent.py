from models.state import RCAState
# from services.openai_service import (get_llm)
from services.llm_service import (get_llm)

llm = get_llm()

def recommendation_agent(
        state: RCAState
):
    prompt = f"""
    RCA:

    {state['root_cause']}


    Generate:

    1 Corrective Action

    2 Preventive Action

    3 Priority

    """
    
    response = llm.invoke(
        prompt
    )

    state["recommendation"] = response.content

    return state