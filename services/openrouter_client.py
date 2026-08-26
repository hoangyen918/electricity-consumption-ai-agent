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


def chat_completion(messages: list, tools: list | None = None, tool_choice: str = "auto"):
    """
    Gọi chat completion qua OpenRouter, có retry với exponential backoff.
    Trả về response object nguyên bản của OpenAI SDK (có .choices, .usage).
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": SETTINGS.chat_model,
                "messages": messages,
                "extra_headers": {
                    "HTTP-Referer": SETTINGS.app_referer,
                    "X-Title": SETTINGS.app_title,
                },
            }
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
