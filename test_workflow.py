# from graph.workflow import graph

# def test_rca():
#     result = graph.invoke(
#         {
#             "question": "why did compressor trip?",
#             "equipment_id": "K101"
#         }
#     )
#     assert result is not None
    
#     assert "root_cause" in result

from ollama import Client

client = Client(host="http://localhost:11434")

response = client.embed(
    model="nomic-embed-text",
    input="hello"
)

print(response)