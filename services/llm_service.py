# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
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