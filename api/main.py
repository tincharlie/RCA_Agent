from fastapi import FastAPI
from graph import workflow
from schema.request_response import (RCARequest, RCAResponse)
from monitoring.prometheus import (REQUEST_COUNT, SUCCESS_COUNT, FAILURE_COUNT)

app = FastAPI(
    title="Agentic RCA Platform"
)

@app.get("/health")
def health():
    return{"status": "healthy"}

@app.post("/rca", response_model= RCAResponse)
def run_rca(request: RCARequest):
    REQUEST_COUNT.inc()

    try:
        result = workflow.graph.invoke(
            {
                "question": request.question,
                "equipment_id": request.equipment_id
            }
        )

        SUCCESS_COUNT.inc()

        return RCAResponse(
            equipment_id=request.equipment_id,
            root_cause= result["roor_cause"],
            confidence= result["confidence"],
            recommendation= result["recommendation"],
            report= result["report"]
        )
    except Exception as e:
        FAILURE_COUNT.inc()
        raise e
    