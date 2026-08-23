# from fastapi import FastAPI
# from graph.workflow import graph
# from schema.request_response import (RCARequest, RCAResponse)
# from monitoring.prometheus import (REQUEST_COUNT, SUCCESS_COUNT, FAILURE_COUNT)

# app = FastAPI(
#     title="Agentic RCA Platform"
# )

# @app.get("/health")
# def health():
#     return{"status": "healthy"}

# @app.post("/rca", response_model= RCAResponse)
# def run_rca(request: RCARequest):
#     REQUEST_COUNT.inc()

#     try:
#         result = graph.invoke(
#             {
#                 "question": request.question,
#                 "equipment_id": request.equipment_id
#             }
#         )

#         print("GRAPH RESULT:", result)
#         SUCCESS_COUNT.inc()

#         return RCAResponse(
#             equipment_id=request.equipment_id,
#             root_cause=result.get("root_cause", ""),
#             confidence=result.get("confidence", 0.0),
#             recommendation=result.get("recommendation", ""),
#             report=result.get("report", "")
#         )
#     except Exception as e:
#         FAILURE_COUNT.inc()
#         raise e
    
from graph.workflow import graph

def main():
    question = input("Enter your question: ")
    equipment_id = input("Enter equipment ID: ")

    result = graph.invoke(
        {
            "question": question,
            "equipment_id": equipment_id
        }
    )

    print("\n===== RCA Result =====")
    print("Equipment ID :", equipment_id)
    print("Root Cause   :", result.get("root_cause"))
    print("Confidence   :", result.get("confidence"))
    print("Recommendation:")
    print(result.get("recommendation"))
    print("\nReport:")
    print(result.get("report"))

if __name__ == "__main__":
    main()