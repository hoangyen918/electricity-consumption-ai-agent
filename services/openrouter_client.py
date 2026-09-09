"""Chat completion client qua CHAT_BASE_URL/CHAT_API_KEY (mặc định fallback về
openrouter_base_url/openrouter_api_key cũ nếu 2 biến này chưa set — xem
config/settings.py::_fallback_chat_endpoint), có retry cho lỗi tạm thời."""
import time
from openai import OpenAI
from config.settings import SETTINGS

# timeout NGẮN hơn default SDK (mặc định openai-python là 600s) — endpoint chat mới
# (proxy tunnel) có lúc trả lỗi/hang chậm hơn nhiều so với OpenRouter thật; không để
# UI treo cả 10 phút chờ 1 request duy nhất trước khi báo lỗi rõ ràng cho người dùng.
CHAT_CLIENT_TIMEOUT_SECONDS = 60.0

_client = OpenAI(
    base_url=SETTINGS.chat_base_url,
    api_key=SETTINGS.chat_api_key,
    timeout=CHAT_CLIENT_TIMEOUT_SECONDS,
)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def chat_completion(
    messages: list,
    tools: list | None = None,
    tool_choice: str = "auto",
    model: str | None = None,
    temperature: float | None = None,
):
    """
    Gọi chat completion qua CHAT_BASE_URL, có retry với exponential backoff.
    Trả về response object nguyên bản của OpenAI SDK (có .choices, .usage).

    model: override model dùng cho lần gọi này (vd LLM-as-a-Judge dùng model khác
    agent chính). Mặc định None -> dùng SETTINGS.chat_model như cũ, không đổi hành vi
    hiện có ở bất kỳ nơi nào khác đang gọi hàm này.

    temperature: override nhiệt độ sampling cho lần gọi này (vd LLM-as-a-Judge dùng
    0.0 để chấm điểm nhất quán). Mặc định None -> KHÔNG truyền tham số này lên API,
    dùng default của model/provider như cũ, không đổi hành vi agent chính.

    stream=False LUÔN được truyền tường minh — endpoint tunnel đã verify thật: nếu
    thiếu field "stream" trong request body, server mặc định trả SSE streaming (data:
    {...}\\n\\n từng dòng) thay vì 1 JSON object hoàn chỉnh, khiến OpenAI SDK parse
    response sai/lỗi. OpenRouter (endpoint cũ) không có vấn đề này, nhưng set tường
    minh ở đây không đổi hành vi OpenRouter (stream=False vốn đã là default ngầm của
    SDK khi không truyền gì).
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model or SETTINGS.chat_model,
                "messages": messages,
                "stream": False,
                "extra_headers": {
                    "HTTP-Referer": SETTINGS.app_referer,
                    "X-Title": SETTINGS.app_title,
                },
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            return _client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(f"  Chat completion failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    # Lỗi rõ ràng, không im lặng treo — RuntimeError này propagate lên tới
    # agent/loop.py -> Agent.ask() -> app.py's try/except (st.error(...)), không crash
    # toàn bộ Streamlit app. last_error giữ nguyên message gốc (timeout/5xx/... đọc
    # được ngay) để dễ chẩn đoán chứ không che giấu nguyên nhân thật.
    raise RuntimeError(
        f"Chat completion failed after {MAX_RETRIES} attempts calling {SETTINGS.chat_base_url}: {last_error}"
    )
