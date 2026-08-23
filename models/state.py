from typing import TypedDict
from typing import List
from typing import Dict


class RCAState(TypedDict):
    question: str
    equipment_id: str
    sensor_data: Dict
    alarm_data: List[str]
    incident_history: List[Dict]
    root_cause: str
    confidence: float
    recommendation: str
    report: str
    
