from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    chat_model: str = "qwen/qwen3.6-27b"
    # chat_base_url/chat_api_key: endpoint RIÊNG cho chat_completion(), tách khỏi
    # openrouter_base_url/openrouter_api_key (vẫn dùng cho embedding, xem
    # rag/embeddings.py — không đụng gì ở đó). Mặc định rỗng "" — nếu .env KHÔNG set
    # CHAT_BASE_URL/CHAT_API_KEY, _fallback_chat_endpoint() bên dưới tự copy lại giá
    # trị openrouter_* cũ, giữ NGUYÊN hành vi hiện tại (backward-compatible, không
    # breaking cho ai chưa cập nhật .env).
    chat_base_url: str = ""
    chat_api_key: str = ""
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

    @model_validator(mode="after")
    def _fallback_chat_endpoint(self) -> "Settings":
        if not self.chat_base_url:
            self.chat_base_url = self.openrouter_base_url
        if not self.chat_api_key:
            self.chat_api_key = self.openrouter_api_key
        return self

SETTINGS = Settings()
