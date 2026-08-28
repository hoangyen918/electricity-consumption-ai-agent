from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    chat_model: str = "qwen/qwen3.6-27b"
    embedding_model: str = "openai/text-embedding-3-small"
    chroma_path: str = "storage/chroma"
    chroma_collection: str = "electricity_kb"
    max_agent_iterations: int = 5
    rag_top_k: int = 10
    app_referer: str = "http://localhost:8501"
    app_title: str = "Electricity Consumption AI Agent"
    tei_embedding_url: str = ""
    tei_reranker_url: str = ""
    tei_api_key: str = ""

SETTINGS = Settings()
