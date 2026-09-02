# import json
# import os
# from typing import Any, Dict, List, TypedDict

# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain_google_genai import (
#     ChatGoogleGenerativeAI,
#     GoogleGenerativeAIEmbeddings,
# )
# from langgraph.graph import END, START, StateGraph

# # Disable LangSmith tracing to prevent unnecessary network logging
# os.environ["LANGCHAIN_TRACING_V2"] = "false"

# load_dotenv()


# # ==========================================
# # 1. STATE DEFINITION
# # ==========================================
# class RCAState(TypedDict, total=False):
#     question: str
#     equipment_id: str
#     sensor_data: Dict[str, Any]
#     alarm_data: List[str]
#     incident_history: List[Dict[str, Any]]
#     rag_context: str
#     root_cause: str
#     confidence: float
#     recommendation: str
#     report: str


# # ==========================================
# # 2. VECTOR DB & MODEL SETUP
# # ==========================================
# print("Setting up embedding model and vector database...")

# # Clean model string without 'models/' prefix prevents double-prefixing in the SDK
# # Change this line:
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="text-embedding-004",  # Updated from text-embedding-001
#     google_api_key=os.getenv("GEMINI_API_KEY"),
# )

# persist_directory = "./chroma_db"

# vector_db = Chroma(
#     collection_name="rca_manuals",
#     embedding_function=embeddings,
#     persist_directory=persist_directory,
# )

# if vector_db._collection.count() == 0:
#     print("Populating persistent database...")
#     manual_docs = [
#         Document(
#             page_content=(
#                 "High vibration and high bearing temperature indicate bearing"
#                 " failure."
#             )
#         ),
#         Document(
#             page_content="Low discharge pressure can be caused by worn bearings."
#         ),
#         Document(
#             page_content=(
#                 "Excessive noise and shaft misalignment can lead to premature"
#                 " bearing wear."
#             )
#         ),
#         Document(
#             page_content=(
#                 "High motor current may indicate pump overloading or internal"
#                 " mechanical damage."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Low suction pressure may be caused by clogged filters or air"
#                 " leakage in the suction line."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Frequent seal failures can result from improper installation"
#                 " or shaft misalignment."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Increased vibration at specific frequencies often indicates"
#                 " imbalance in rotating parts."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Overheating of motor windings can be caused by poor"
#                 " ventilation or electrical faults."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Reduced flow rate may result from impeller wear or blockage in"
#                 " the system."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Oil contamination in bearings can lead to increased friction"
#                 " and eventual failure."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Cavitation noise and vibration occur due to low suction head or"
#                 " vapor formation in the fluid."
#             )
#         ),
#         Document(
#             page_content=(
#                 "Loose mounting bolts can cause excessive vibration and"
#                 " misalignment issues."
#             )
#         ),
#     ]
#     vector_db.add_documents(manual_docs)
#     print("Data saved to disk at:", persist_directory)
# else:
#     print(
#         f"Loaded existing database with {vector_db._collection.count()} items."
#     )

# # Active chat model setup
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GEMINI_API_KEY"),
#     temperature=0,
# )


# def get_equipment_data(equipment_id: str):
#     return {
#         "vibration": 10.8,
#         "bearing_temp": 112,
#         "suction_pressure": 4.2,
#         "discharge_pressure": 6.1,
#         "flow": 50,
#     }


# def get_historical_incidents(equipment_id: str):
#     return [
#         {
#             "incident_id": "INC-101",
#             "issue": "High vibration detected during Q3 audit",
#             "resolution": "Re-aligned shaft",
#         },
#         {
#             "incident_id": "INC-204",
#             "issue": "Bearing temp spike",
#             "resolution": "Replaced bearing oil",
#         },
#     ]


# # ==========================================
# # 3. AGENT NODE DEFINITIONS
# # ==========================================
# def rag_agent(state: RCAState) -> Dict:
#     print("[Agent] Running RAG Retrieval...")
#     docs = vector_db.similarity_search(state["question"], k=3)
#     context = "\n".join([doc.page_content for doc in docs])
#     return {"rag_context": context}


# def sensor_agent(state: RCAState) -> Dict:
#     print("[Agent] Fetching Live Sensor Data...")
#     sensor_data = get_equipment_data(state["equipment_id"])
#     return {"sensor_data": sensor_data}


# def alarm_agent(state: RCAState) -> Dict:
#     print("[Agent] Checking Active Alarms...")
#     alarms = ["HIGH_VIBRATION", "HIGH_BEARING_TEMP", "LOW_DISCHARGE_PRESSURE"]
#     return {"alarm_data": alarms}


# def history_agent(state: RCAState) -> Dict:
#     print("[Agent] Querying Historical Incidents...")
#     incidents = get_historical_incidents(state["equipment_id"])
#     return {"incident_history": incidents}


# def rca_agent(state: RCAState) -> Dict:
#     print("[Agent] Analyzing Root Cause with LLM...")
#     prompt = f"""
#     You are a senior reliability engineer conducting Root Cause Analysis (RCA).

#     User Question: {state.get('question', '')}
#     Equipment ID: {state.get('equipment_id', '')}

#     Manual & Documentation Context:
#     {state.get('rag_context', 'N/A')}

#     Live Sensor Data:
#     {json.dumps(state.get('sensor_data', {}))}

#     Triggered Alarms:
#     {state.get('alarm_data', [])}

#     Historical Incidents:
#     {json.dumps(state.get('incident_history', []))}

#     Respond ONLY with a valid JSON object matching this key structure:
#     {{
#         "root_cause": "Detailed analysis of root cause",
#         "confidence": 0.95
#     }}
#     """
#     try:
#         response = llm.invoke(prompt)
#         content = (
#             response.content.strip()
#             .replace("```json", "")
#             .replace("```", "")
#             .strip()
#         )
#         data = json.loads(content)
#         return {
#             "root_cause": data.get("root_cause", response.content),
#             "confidence": float(data.get("confidence", 0.90)),
#         }
#     except Exception:
#         response = llm.invoke(prompt)
#         return {"root_cause": response.content, "confidence": 0.90}


# def recommendation_agent(state: RCAState) -> Dict:
#     print("[Agent] Generating Actionable Recommendations...")
#     prompt = f"""
#     Based on the following Root Cause Analysis:
#     {state.get('root_cause', '')}

#     Generate structured recommendations:
#     1. Corrective Action
#     2. Preventive Action
#     3. Priority Level (High, Medium, Low)
#     """
#     response = llm.invoke(prompt)
#     return {"recommendation": response.content}


# def report_agent(state: RCAState) -> Dict:
#     print("[Agent] Compiling Final RCA Report...")
#     report = f"""
# ==================================================
#               ROOT CAUSE ANALYSIS REPORT
# ==================================================
# Equipment ID : {state.get('equipment_id', 'N/A')}
# Issue Query  : {state.get('question', 'N/A')}

# ROOT CAUSE:
# {state.get('root_cause', 'N/A')}

# CONFIDENCE SCORE: {state.get('confidence', 0.0):.2f}

# RECOMMENDATION:
# {state.get('recommendation', 'N/A')}
# ==================================================
# """
#     return {"report": report}


# # ==========================================
# # 4. WORKFLOW GRAPH CONSTRUCTION
# # ==========================================
# workflow = StateGraph(RCAState)

# workflow.add_node("rag_agent", rag_agent)
# workflow.add_node("sensor_agent", sensor_agent)
# workflow.add_node("alarm_agent", alarm_agent)
# workflow.add_node("history_agent", history_agent)
# workflow.add_node("rca_agent", rca_agent)
# workflow.add_node("recommendation_agent", recommendation_agent)
# workflow.add_node("report_agent", report_agent)

# workflow.add_edge(START, "rag_agent")
# workflow.add_edge(START, "sensor_agent")
# workflow.add_edge(START, "alarm_agent")
# workflow.add_edge(START, "history_agent")

# workflow.add_edge("rag_agent", "rca_agent")
# workflow.add_edge("sensor_agent", "rca_agent")
# workflow.add_edge("alarm_agent", "rca_agent")
# workflow.add_edge("history_agent", "rca_agent")

# workflow.add_edge("rca_agent", "recommendation_agent")
# workflow.add_edge("recommendation_agent", "report_agent")
# workflow.add_edge("report_agent", END)

# app = workflow.compile()

# # ==========================================
# # 5. EXECUTION ENTRYPOINT
# # ==========================================
# if __name__ == "__main__":
#     print("\nStarting RCA Graph Execution...\n")

#     input_state = {
#         "question": (
#             "Why is the pump experiencing high vibration and temperature"
#             " spikes?"
#         ),
#         "equipment_id": "PUMP-K101",
#     }

#     final_state = app.invoke(input_state)
#     print(final_state["report"])


# import os
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# from langchain_core.documents import Document
# from langchain_core.embeddings import Embeddings
# from langchain_chroma import Chroma

# # 1. Ensure .env file is loaded into environment variables
# load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise ValueError("GEMINI_API_KEY environment variable is not set. Check your .env file.")

# class DirectGeminiEmbeddings(Embeddings):
#     def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
#         self.client = genai.Client(api_key=api_key)
#         self.model = model.replace("models/", "")

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         response = self.client.models.embed_content(
#             model=self.model,
#             contents=texts,
#             config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
#         )
#         return [item.values for item in response.embeddings]

#     def embed_query(self, text: str) -> list[float]:
#         response = self.client.models.embed_content(
#             model=self.model,
#             contents=text,
#             config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
#         )
#         return response.embeddings[0].values


# # 2. Instantiate embeddings
# embeddings = DirectGeminiEmbeddings(
#     api_key=api_key,
#     model="gemini-embedding-001"
# )

# # 3. Vector DB persistence setup
# persist_directory = "./chroma_db"

# vector_db = Chroma(
#     collection_name="rca_manuals",
#     embedding_function=embeddings,
#     persist_directory=persist_directory,
# )

# # 4. Populate database if empty
# if vector_db._collection.count() == 0:
#     print("Populating persistent database...")
#     manual_docs = [
#         Document(
#             page_content="High vibration and high bearing temperature indicate bearing failure."
#         ),
#         Document(
#             page_content="Low discharge pressure can be caused by worn bearings."
#         ),
#     ]
#     vector_db.add_documents(manual_docs)
#     print("Data saved to disk at:", persist_directory)
# else:
#     print(f"Loaded existing database with {vector_db._collection.count()} items.")

# # 5. Query
# results = vector_db.similarity_search("vibration issues", k=1)
# print("Search Result:", results[0].page_content)


import json
import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

# Disable LangSmith tracing to prevent unnecessary network logging
os.environ["LANGCHAIN_TRACING_V2"] = "false"

load_dotenv()

# Verify API key availability
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Check your .env file.")


# ==========================================
# 1. CUSTOM GEMINI EMBEDDINGS (google-genai)
# ==========================================
class DirectGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model.replace("models/", "")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [item.values for item in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class RCAState(TypedDict, total=False):
    question: str
    equipment_id: str
    sensor_data: Dict[str, Any]
    alarm_data: List[str]
    incident_history: List[Dict[str, Any]]
    rag_context: str
    root_cause: str
    confidence: float
    recommendation: str
    report: str


# ==========================================
# 3. VECTOR DB & MODEL SETUP
# ==========================================
print("Setting up embedding model and vector database...")

embeddings = DirectGeminiEmbeddings(
    api_key=api_key,
    model="gemini-embedding-001"
)

persist_directory = "./chroma_db"

vector_db = Chroma(
    collection_name="rca_manuals",
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

if vector_db._collection.count() == 0:
    print("Populating persistent database...")
    manual_docs = [
        Document(
            page_content=(
                "High vibration and high bearing temperature indicate bearing"
                " failure."
            )
        ),
        Document(
            page_content="Low discharge pressure can be caused by worn bearings."
        ),
        Document(
            page_content=(
                "Excessive noise and shaft misalignment can lead to premature"
                " bearing wear."
            )
        ),
        Document(
            page_content=(
                "High motor current may indicate pump overloading or internal"
                " mechanical damage."
            )
        ),
        Document(
            page_content=(
                "Low suction pressure may be caused by clogged filters or air"
                " leakage in the suction line."
            )
        ),
        Document(
            page_content=(
                "Frequent seal failures can result from improper installation"
                " or shaft misalignment."
            )
        ),
        Document(
            page_content=(
                "Increased vibration at specific frequencies often indicates"
                " imbalance in rotating parts."
            )
        ),
        Document(
            page_content=(
                "Overheating of motor windings can be caused by poor"
                " ventilation or electrical faults."
            )
        ),
        Document(
            page_content=(
                "Reduced flow rate may result from impeller wear or blockage in"
                " the system."
            )
        ),
        Document(
            page_content=(
                "Oil contamination in bearings can lead to increased friction"
                " and eventual failure."
            )
        ),
        Document(
            page_content=(
                "Cavitation noise and vibration occur due to low suction head or"
                " vapor formation in the fluid."
            )
        ),
        Document(
            page_content=(
                "Loose mounting bolts can cause excessive vibration and"
                " misalignment issues."
            )
        ),
    ]
    vector_db.add_documents(manual_docs)
    print("Data saved to disk at:", persist_directory)
else:
    print(
        f"Loaded existing database with {vector_db._collection.count()} items."
    )

# Active chat model setup
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0,
)


def get_equipment_data(equipment_id: str):
    return {
        "vibration": 10.8,
        "bearing_temp": 112,
        "suction_pressure": 4.2,
        "discharge_pressure": 6.1,
        "flow": 50,
    }


def get_historical_incidents(equipment_id: str):
    return [
        {
            "incident_id": "INC-101",
            "issue": "High vibration detected during Q3 audit",
            "resolution": "Re-aligned shaft",
        },
        {
            "incident_id": "INC-204",
            "issue": "Bearing temp spike",
            "resolution": "Replaced bearing oil",
        },
    ]


# ==========================================
# 4. AGENT NODE DEFINITIONS
# ==========================================
def rag_agent(state: RCAState) -> Dict:
    print("[Agent] Running RAG Retrieval...")
    docs = vector_db.similarity_search(state["question"], k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return {"rag_context": context}


def sensor_agent(state: RCAState) -> Dict:
    print("[Agent] Fetching Live Sensor Data...")
    sensor_data = get_equipment_data(state["equipment_id"])
    return {"sensor_data": sensor_data}


def alarm_agent(state: RCAState) -> Dict:
    print("[Agent] Checking Active Alarms...")
    alarms = ["HIGH_VIBRATION", "HIGH_BEARING_TEMP", "LOW_DISCHARGE_PRESSURE"]
    return {"alarm_data": alarms}


def history_agent(state: RCAState) -> Dict:
    print("[Agent] Querying Historical Incidents...")
    incidents = get_historical_incidents(state["equipment_id"])
    return {"incident_history": incidents}


def rca_agent(state: RCAState) -> Dict:
    print("[Agent] Analyzing Root Cause with LLM...")
    prompt = f"""
    You are a senior reliability engineer conducting Root Cause Analysis (RCA).

    User Question: {state.get('question', '')}
    Equipment ID: {state.get('equipment_id', '')}

    Manual & Documentation Context:
    {state.get('rag_context', 'N/A')}

    Live Sensor Data:
    {json.dumps(state.get('sensor_data', {}))}

    Triggered Alarms:
    {state.get('alarm_data', [])}

    Historical Incidents:
    {json.dumps(state.get('incident_history', []))}

    Respond ONLY with a valid JSON object matching this key structure:
    {{
        "root_cause": "Detailed analysis of root cause",
        "confidence": 0.95
    }}
    """
    try:
        response = llm.invoke(prompt)
        content = (
            response.content.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        data = json.loads(content)
        return {
            "root_cause": data.get("root_cause", response.content),
            "confidence": float(data.get("confidence", 0.90)),
        }
    except Exception:
        response = llm.invoke(prompt)
        return {"root_cause": response.content, "confidence": 0.90}


def recommendation_agent(state: RCAState) -> Dict:
    print("[Agent] Generating Actionable Recommendations...")
    prompt = f"""
    Based on the following Root Cause Analysis:
    {state.get('root_cause', '')}

    Generate structured recommendations:
    1. Corrective Action
    2. Preventive Action
    3. Priority Level (High, Medium, Low)
    """
    response = llm.invoke(prompt)
    return {"recommendation": response.content}


def report_agent(state: RCAState) -> Dict:
    print("[Agent] Compiling Final RCA Report...")
    report = f"""
==================================================
              ROOT CAUSE ANALYSIS REPORT
==================================================
Equipment ID : {state.get('equipment_id', 'N/A')}
Issue Query  : {state.get('question', 'N/A')}

ROOT CAUSE:
{state.get('root_cause', 'N/A')}

CONFIDENCE SCORE: {state.get('confidence', 0.0):.2f}

RECOMMENDATION:
{state.get('recommendation', 'N/A')}
==================================================
"""
    return {"report": report}


# ==========================================
# 5. WORKFLOW GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(RCAState)

workflow.add_node("rag_agent", rag_agent)
workflow.add_node("sensor_agent", sensor_agent)
workflow.add_node("alarm_agent", alarm_agent)
workflow.add_node("history_agent", history_agent)
workflow.add_node("rca_agent", rca_agent)
workflow.add_node("recommendation_agent", recommendation_agent)
workflow.add_node("report_agent", report_agent)

workflow.add_edge(START, "rag_agent")
workflow.add_edge(START, "sensor_agent")
workflow.add_edge(START, "alarm_agent")
workflow.add_edge(START, "history_agent")

workflow.add_edge("rag_agent", "rca_agent")
workflow.add_edge("sensor_agent", "rca_agent")
workflow.add_edge("alarm_agent", "rca_agent")
workflow.add_edge("history_agent", "rca_agent")

workflow.add_edge("rca_agent", "recommendation_agent")
workflow.add_edge("recommendation_agent", "report_agent")
workflow.add_edge("report_agent", END)

app = workflow.compile()

# ==========================================
# 6. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("\nStarting RCA Graph Execution...\n")

    input_state = {
        "question": (
            "Why is the pump experiencing high vibration and temperature"
            " spikes?"
        ),
        "equipment_id": "PUMP-K101",
    }

    final_state = app.invoke(input_state)
    print(final_state["report"])