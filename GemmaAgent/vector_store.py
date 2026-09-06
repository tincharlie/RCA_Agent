from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import GEMINI_API_KEY


class DirectGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model.replace("models/", "")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [item.values for item in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values


def init_vector_db(persist_directory: str = "./chroma_db") -> Chroma:
    embeddings = DirectGeminiEmbeddings(api_key=GEMINI_API_KEY)
    vector_db = Chroma(
        collection_name="rca_manuals",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    if vector_db._collection.count() == 0:
        manual_docs = [
            Document(page_content="High vibration and high bearing temperature indicate bearing failure."),
            Document(page_content="Low discharge pressure can be caused by worn bearings."),
            Document(page_content="Excessive noise and shaft misalignment can lead to premature bearing wear."),
            Document(page_content="High motor current may indicate pump overloading or internal mechanical damage."),
            Document(page_content="Low suction pressure may be caused by clogged filters or air leakage in the suction line."),
            Document(page_content="Frequent seal failures can result from improper installation or shaft misalignment."),
            Document(page_content="Increased vibration at specific frequencies often indicates imbalance in rotating parts."),
            Document(page_content="Overheating of motor windings can be caused by poor ventilation or electrical faults."),
            Document(page_content="Reduced flow rate may result from impeller wear or blockage in the system."),
            Document(page_content="Oil contamination in bearings can lead to increased friction and eventual failure."),
            Document(page_content="Cavitation noise and vibration occur due to low suction head or vapor formation in the fluid."),
            Document(page_content="Loose mounting bolts can cause excessive vibration and misalignment issues."),
        ]
        vector_db.add_documents(manual_docs)

    return vector_db


vector_db = init_vector_db()