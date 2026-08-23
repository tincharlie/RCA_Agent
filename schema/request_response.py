from pydantic import BaseModel

class RCARequest(BaseModel):
    question: str
    equipment_id: str

class RCAResponse(BaseModel):
    equipment_id: str
    root_cause: str
    confidence: float
    recommendation: str
    report: str

