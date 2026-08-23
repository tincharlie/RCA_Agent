from models.state import RCAState
from services.mongodb_service import (get_incident_collection)

def history_agent(state: RCAState):
    collection = (
        get_incident_collection()
    )

    incidents = list(
        collection.find(
            {"equipment_id": state["equipment_id"]}
        ).limit(5)
    )
    
    state["incident_hisory"] = incidents

    return state