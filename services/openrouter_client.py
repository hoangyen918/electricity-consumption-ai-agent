"""Chat completion client qua OpenRouter, có retry cho lỗi tạm thời."""
import time
from openai import OpenAI
from config.settings import SETTINGS

_client = OpenAI(
    base_url=SETTINGS.openrouter_base_url,
    api_key=SETTINGS.openrouter_api_key,
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
    Gọi chat completion qua OpenRouter, có retry với exponential backoff.
    Trả về response object nguyên bản của OpenAI SDK (có .choices, .usage).

    model: override model dùng cho lần gọi này (vd LLM-as-a-Judge dùng model khác
    agent chính). Mặc định None -> dùng SETTINGS.chat_model như cũ, không đổi hành vi
    hiện có ở bất kỳ nơi nào khác đang gọi hàm này.

    temperature: override nhiệt độ sampling cho lần gọi này (vd LLM-as-a-Judge dùng
    0.0 để chấm điểm nhất quán). Mặc định None -> KHÔNG truyền tham số này lên API,
    dùng default của model/provider như cũ, không đổi hành vi agent chính.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model or SETTINGS.chat_model,
                "messages": messages,
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
    raise RuntimeError(f"Chat completion failed after {MAX_RETRIES} attempts: {last_error}")
