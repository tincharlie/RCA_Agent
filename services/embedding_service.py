# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from config.settings import settings

print("OLLAMA_EMBED_MODEL =", settings.OLLAMA_EMBED_MODEL)
embeddings = OllamaEmbeddings(
    model=settings.OLLAMA_EMBED_MODEL
)

def get_embeddings():
    return embeddings