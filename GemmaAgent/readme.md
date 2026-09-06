rca_project/
│
├── .env # API keys and environment configuration
├── config.py # App-wide settings and clients
├── state.py # RCAState TypedDict and Pydantic schemas
├── vector_store.py # Chroma DB setup and DirectGeminiEmbeddings
├── nodes.py # All LangGraph agent node functions
├── router.py # Route logic and Graph compilation
├── api.py # FastAPI backend endpoints
└── app.py # Streamlit dashboard frontend

uvicorn api:api --reload --port 8000

streamlit run app.py
