from models.state import RCAState
from services.pi_service import get_equipment_data

def sensor_agent(state: RCAState):
    equipment_id = (state["equipment_id"])

    sensor_data = (get_equipment_data(equipment_id))

    state["sensor_data"] = (sensor_data)

    return state