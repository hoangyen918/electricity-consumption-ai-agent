"""
Tính chi phí ước tính dựa trên bảng giá cấu hình sẵn.
Giá lấy từ openrouter.ai, xác nhận lần cuối vào ngày PRICING_LAST_VERIFIED.
Lưu ý: OpenRouter định tuyến qua nhiều nhà cung cấp khác nhau cho cùng 1 model,
giá thực tế trả có thể chênh nhẹ so với bảng này. Để đối chiếu chính xác tuyệt
đối cho 1 request cụ thể, dùng GET https://openrouter.ai/api/v1/generation?id=<response_id>
(id lấy từ response.id của lần gọi chat_completion tương ứng).
"""

PRICING_LAST_VERIFIED = "2026-08-26"

MODEL_PRICING = {
    "qwen/qwen3.6-27b": {
        "input_per_million": 0.30,
        "output_per_million": 2.00,
    },
    "openai/text-embedding-3-small": {
        "input_per_million": 0.02,
        "output_per_million": 0.0,
    },
}


def calculate_chat_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Trả 0.0 nếu model không có trong bảng giá, không raise lỗi."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    cost = (
        input_tokens * pricing["input_per_million"] / 1_000_000
        + output_tokens * pricing["output_per_million"] / 1_000_000
    )
    return round(cost, 8)


def calculate_embedding_cost(model: str, tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    cost = tokens * pricing["input_per_million"] / 1_000_000
    return round(cost, 8)
