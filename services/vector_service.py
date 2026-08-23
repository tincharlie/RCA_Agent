# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from services.embedding_service import get_embeddings

embeddings = get_embeddings()

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def get_vector_db():
    return vector_db