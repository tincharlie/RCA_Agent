from models.state import RCAState

def report_agent(
        state: RCAState
):
    
    report = f"""
    RCA REPORT

    Equipment:
    {state["equipment_id"]}

    Question:
    {state["question"]}

    Root Cause:
    {state["root_cause"]}

    Confidence:
    {state["confidence"]}

    Recommendation:
    {state["recommendation"]}
    """

    state["report"] = report

    return state

