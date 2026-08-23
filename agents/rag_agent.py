from services.vector_service import get_vector_db


db = get_vector_db()

def rag_agent(state):
    docs= db.similarity_search(
        state["question"],
        k = 3
    )

    context = "\n".join(
        [
            doc.page_content for doc in docs
        ]
    )

    state["rag_context"] = context
    return state


# from models.state import RCAState
# from services.mongodb_service import (get_manual_collection)

# collection = get_manual_collection()

# def rag_agent(state: RCAState):
#     query = state["question"]

#     results = collection.aggregate(
#         [
#             {
#                 "$vectorSearch": {
#                     "index":"manual_vector_index",
#                     "path": "embedding",
#                     "queryVector": [0.1] * 1536,
#                     "numCandidates":100,
#                     "limit":5
#                 }
#             }
#         ]
#     )
    
#     context = []

#     for doc in results:

#         context.append(
#             doc["content"]
#         )

#     state["rag_context"] = (
#         "\n".join(context)
#     )

#     return state

