from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from router import app_graph
from state import RCAState

api = FastAPI(title="RCA Agent Microservice", version="1.0.0")


class RCARequest(BaseModel):
    equipment_id: str = Field(..., example="PUMP-K204")
    question: str = Field(..., example="Why is the pump experiencing high vibration?")


class RCAResponse(BaseModel):
    equipment_id: str
    root_cause: str
    confidence: float
    faithfulness_score: float
    is_hallucinated: bool
    recommendation: dict
    report: str


@api.post("/api/v1/rca", response_model=RCAResponse)
async def run_rca_analysis(payload: RCARequest):
    input_state: RCAState = {
        "question": payload.question,
        "equipment_id": payload.equipment_id,
    }

    try:
        final_state = await app_graph.ainvoke(input_state)
        return RCAResponse(
            equipment_id=payload.equipment_id,
            root_cause=final_state.get("root_cause", ""),
            confidence=final_state.get("confidence", 0.0),
            faithfulness_score=final_state.get("faithfulness_score", 0.0),
            is_hallucinated=final_state.get("is_hallucinated", False),
            recommendation=final_state.get("recommendation", {}),
            report=final_state.get("report", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)