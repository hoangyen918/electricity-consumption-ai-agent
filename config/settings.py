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
    # base_url: endpoint embedding LOCAL (Ollama-native /api/embed, KHÁC format
    # OpenAI-compatible) — đổi 2026-09-22 sang bge-m3 local vì OPENROUTER_API_KEY hết
    # hạn (401) VÀ để không tốn phí. Rỗng "" nếu chưa cấu hình local embedding.
    embedding_base_url: str = ""
    chroma_path: str = "storage/chroma"
    chroma_collection: str = "electricity_kb"
    max_agent_iterations: int = 5
    # Request Budget (Phần 1/3, 2026-09) — ngân sách TỔNG cho 1 lượt hỏi-đáp, ĐỘC LẬP
    # với max_agent_iterations (chặn theo THỜI GIAN/TOKEN/CHI PHÍ/SỐ TOOL CALL, không
    # phải số vòng lặp). Ngưỡng dựa trên đo THẬT logs/agent_trace.jsonl (1384 request
    # thật, đo 2026-09-18) + evals/results/eval_detail_*.json — KHÔNG phải số tròn tự
    # nghĩ, là HEURISTIC ban đầu, cần điều chỉnh khi có thêm dữ liệu:
    #   - max_request_seconds: latency thô CAO NHẤT quan sát được là ~2153s, nhưng đây
    #     là artifact phiên bị tạm dừng thật (vd "Dataset có bao nhiêu dòng?" — câu hỏi
    #     đơn giản không thể mất 24 phút xử lý thật) — dùng phân phối THỰC (median=17.7s,
    #     p95=74.4s, p99=251.2s) làm cơ sở thay vì max thô bị nhiễu, chọn 300s (~1.2x
    #     p99) đủ rộng cho case chậm thật, vẫn chặn được vòng lặp treo thật sự.
    #   - max_request_tokens: max thô THẬT (không bị nhiễu như latency, token đếm được
    #     chính xác) = 108,985 (case "Curtailment tháng 3..." lặp lại nhiều lần, p99=
    #     74,039) — chọn 150,000 (~1.4x max thô).
    #   - max_request_cost_usd: max thô THẬT = $0.0355 (p99=$0.0239) — chọn $0.05
    #     (~1.4x max thô).
    #   - max_tool_calls: max thô THẬT = 11 (case "tổng quan về historic demand", 5
    #     iterations/11 tool call — HỢP LỆ, thành công, KHÔNG phải vòng lặp lỗi, xác
    #     nhận 1 iteration có thể chứa NHIỀU tool_calls nên số này PHẢI khác/lớn hơn
    #     max_agent_iterations) — chọn 20 (~1.8x max thô, còn cách xa max_agent_
    #     iterations x nhiều để không chặn nhầm case hợp lệ tương tự).
    max_request_seconds: float = 300.0
    max_request_tokens: int = 150_000
    max_request_cost_usd: float = 0.05
    max_tool_calls: int = 20
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
print(f"SETTINGS.: {SETTINGS.embedding_model}")