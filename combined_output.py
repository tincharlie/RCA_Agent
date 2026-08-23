combined_output.py


load_documents.py
from langchain_core.documents import Document
from services.vector_service import vector_db

docs = [
    Document(
        page_content="""
        High vibration and high bearing temperature indicate bearing failure.
        """
    ),
    Document(
        page_content="""
        Low discharge pressure can be caused by worn bearings.
        """
    ),
    Document(
        page_content="""
        Excessive noise and shaft misalignment can lead to premature bearing wear.
        """
    ),
    Document(
        page_content="""
        High motor current may indicate pump overloading or internal mechanical damage.
        """
    ),
    Document(
        page_content="""
        Low suction pressure may be caused by clogged filters or air leakage in the suction line.
        """
    ),
    Document(
        page_content="""
        Frequent seal failures can result from improper installation or shaft misalignment.
        """
    ),
    Document(
        page_content="""
        Increased vibration at specific frequencies often indicates imbalance in rotating parts.
        """
    ),
    Document(
        page_content="""
        Overheating of motor windings can be caused by poor ventilation or electrical faults.
        """
    ),
    Document(
        page_content="""
        Reduced flow rate may result from impeller wear or blockage in the system.
        """
    ),
    Document(
        page_content="""
        Oil contamination in bearings can lead to increased friction and eventual failure.
        """
    ),
    Document(
        page_content="""
        Cavitation noise and vibration occur due to low suction head or vapor formation in the fluid.
        """
    ),
    Document(
        page_content="""
        Loose mounting bolts can cause excessive vibration and misalignment issues.
        """
    )
]


vector_db.add_documents(docs)

pythonfileextract.py
import os

def extract_python_files(root_folder, output_file):
    skip_folders = {'venv', '__pycache__', '.git'}

    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        for current_folder, dirs, files in os.walk(root_folder):
            # ✅ Modify dirs in-place to SKIP folders (not delete them!)
            dirs[:] = [d for d in dirs if d not in skip_folders]

            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(current_folder, file)

                    relative_path = os.path.relpath(full_path, root_folder)

                    try:
                        with open(full_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                    except Exception as e:
                        content = f"# Error reading file: {e}"

                    outfile.write(f"{relative_path}\n")
                    outfile.write(content)
                    outfile.write("\n\n")


# ✅ Usage
root_directory = r"C:\Users\H538532\Desktop\RCA_Agent"
output_file = "combined_output.py"

extract_python_files(root_directory, output_file)

agents\alarm_agent.py
from models.state import RCAState


def alarm_agent(
        state: RCAState
):
    alarms = [
        "HIGH_VIBRATION",
        "HIGH_BEARING_TEMP",
        "LOW_DISCHARGE_PRESSURE"
    ]

    state["alarm_data"] =(alarms)

    return state

agents\history_agent.py
from models.state import RCAState
from services.mongodb_service import (get_incident_collection)

def history_agents(state: RCAState):
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

agents\rag_agent.py
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



agents\rca_agent.py
from models.state import RCAState
# from services.openai_service import (get_llm)
from services.llm_service import (get_llm)

llm = get_llm()

def rca_agent(
        state: RCAState
):

    prompt = f"""

    You are a senior reliability engineer.

    Question:
    {state['question']}

    Manual Context:
    {state['rag_context']}

    Sensor Data:
    {state['sensor_data']}

    Alarm Data:
    {state['alarm_data']}

    Historical Incidents:
    {state['incident_history']}

    Provide:

    1 Root Cause

    2 Confidence Score

    3 Evidence

    """

    response = llm.invoke(
        prompt
    )

    state["root_cause"] = (
        response.content
    )

    state["confidence"] = 0.92

    return state
    

agents\recommendation_agent.py
from models.state import RCAState
# from services.openai_service import (get_llm)
from services.llm_service import (get_llm)

llm = get_llm()

def recommendation_agent(
        state: RCAState
):
    prompt = f"""
    RCA:

    {state['root_cause']}


    Generate:

    1 Corrective Action

    2 Preventive Action

    3 Priority

    """
    
    response = llm.invoke(
        prompt
    )

    state["recommendation"] = response.content

    return state

agents\report_agent.py
from models.state import RCAState

def report_agent(
        state: RCAState
):
    
    report = f"""
    RCA REPORT

    Equipment:
    {state["equipment_id"]}

    Question:
    {state["question"]}

    Root Cause:
    {state["root_cause"]}

    Confidence:
    {state["confidence"]}

    Recommendation:
    {state["recommendation"]}
    """

    state["report"] = report

    return state



agents\sensor_agent.py
from models.state import RCAState
from services.pi_service import get_equipment_data

def sensor_agent(state: RCAState):
    equipment_id = (state["equipment_id"])

    sensor_data = (get_equipment_data(equipment_id))

    state["sensor_data"] = (sensor_data)

    return state

api\main.py
from fastapi import FastAPI
from graph.workflow import graph
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
        result = graph.invoke(
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
    

config\logging.py

import logging


def setup_logger():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    return logging.getLogger(
        "agentic_rca"
    )


logger = setup_logger()

config\settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str

    MONGO_URI:str
    MONGO_DB: str

    CHROMA_DB_PATH: str
    OLLAMA_MODEL: str
    OLLAMA_EMBED_MODEL: str

    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = ""

    class Config:
        env_file = ".env"
    
settings = Settings()

graph\workflow.py
from langgraph.graph import (StateGraph, END)
from models.state import RCAState
from agents.rag_agent import (rag_agent)
from agents.sensor_agent import (sensor_agent)
from agents.alarm_agent import (alarm_agent)
from agents.history_agent import (history_agent)
from agents.rca_agent import (rca_agent)
from agents.recommendation_agent import (recommendation_agent)
from agents.report_agent import (report_agent)

workflow = StateGraph(
    RCAState
)

workflow.add_node(
    "rag_agent",
    rag_agent
)

workflow.add_node(
    "sensor_agent",
    sensor_agent
)

workflow.add_node(
    "alarm_agent",
    alarm_agent
)

workflow.add_node(
    "history_agent",
    history_agent
)

workflow.add_node(
    "rca_agent",
    rca_agent
)

workflow.add_node(
    "recommendation_agent",
    recommendation_agent
)

workflow.add_node(
    "report_agent",
    report_agent
)

workflow.set_entry_point(
    "rag_agent"
)

workflow.add_edge(
    "rag_agent",
    "sensor_agent"
)


workflow.add_edge(
    "sensor_agent",
    "alarm_agent"
)


workflow.add_edge(
    "alarm_agent",
    "history_agent"
)


workflow.add_edge(
    "history_agent",
    "rca_agent"
)

workflow.add_edge(
    "rca_agent",
    "recommendation_agent"
)


workflow.add_edge(
    "recommendation_agent",
    "report_agent"
)

workflow.add_edge(
    "report_agent",
    END
)

graph = workflow.compile()

middleware\auth.py
from fastapi import Header
from fastapi import HTTPException

def validate_token(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Token")
    return True




models\state.py
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
    


monitoring\prometheus.py
from prometheus_client import Counter

REQUEST_COUNT = Counter(
    "rca_request_total",
    "Total RCA Requests"
)

SUCCESS_COUNT = Counter(
    "rca_success_total",
    "Successful RCA Runs"
)

FAILURE_COUNT = Counter(
    "rca_failure_total",
    "Failed RCA Runs"
)

schema\request_response.py
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



services\embedding_service.py
from langchain_community.embeddings import OllamaEmbeddings
from config.settings import settings

embeddings = OllamaEmbeddings(
    model=settings.OLLAMA_EMBED_MODEL
)

def get_embeddings():
    return embeddings

services\llm_service.py
from langchain_community.chat_models import ChatOllama
# from langchain_openai import ChatOpenAI
from config.settings import settings

# llm = ChatOpenAI(
#     api_key=settings.OPENAI_API_KEY,
#     model = "gpt-4o",
#     temperature = 0
# )

llm = ChatOllama(
    model=settings.OLLAMA_MODEL,
    temperature = 0
)

def get_llm():
    return llm

services\mongodb_service.py
from pymongo import MongoClient
from config.settings import settings


client = MongoClient(
    settings.MONGO_URI
)

db =client[
    settings.MONGO_DB
]

def get_manual_collection():
    return db.manuals

def get_incident_collection():
    return db.incidents



services\openai_service.py
from langchain_openai import ChatOpenAI
from config.settings import settings

llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model = "gpt-4o",
    temperature = 0
)

def get_llm():
    return llm

services\pi_service.py
def get_equipment_data(equipment_id: str):
    return {
        "vibration": 10.8,
        "bearing_temp": 112,
        "suction_pressure": 4.2,
        "discharge_pressure": 6.1,
        "flow": 50
    }

services\postgres_service.py
from sqlalchemy import create_engine

from config.settings import settings


DATABASE_URL = (
    f"postgresql://"
    f"{settings.POSTGRES_USER}:"
    f"{settings.POSTGRES_PASSWORD}@"
    f"{settings.POSTGRES_HOST}:"
    f"{settings.POSTGRES_PORT}/"
    f"{settings.POSTGRES_DB}"
)

engine = create_engine(
    DATABASE_URL
)

services\vector_service.py
from langchain_community.vectorstores import Chroma
from services.embedding_service import get_embeddings

embeddings = get_embeddings()

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def get_vector_db():
    return vector_db

tests\test_workflow.py
from graph.workflow import graph

def test_rca():
    result = graph.invoke(
        {
            "question": "why did compressor trip?",
            "equipment_id": "K101"
        }
    )
    assert result is not None
    
    assert "root_cause" in result

