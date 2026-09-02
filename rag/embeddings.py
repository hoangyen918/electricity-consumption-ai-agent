"""Gọi embedding model qua OpenRouter, có batching và retry."""
import time
from openai import OpenAI
from config.settings import SETTINGS

_client = OpenAI(
    base_url=SETTINGS.openrouter_base_url,
    api_key=SETTINGS.openrouter_api_key,
)

BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # nhân đôi mỗi lần retry: 2s, 4s (chỉ chờ giữa các lần thử, không chờ sau lần cuối)


def _embed_batch(texts: list[str], usage_accumulator: list[int] | None = None) -> list[list[float]]:
    """Gọi API cho 1 batch, có retry với exponential backoff.

    usage_accumulator: nếu truyền vào, token count của LẦN GỌI THÀNH CÔNG được append
    vào đây (không cộng dồn các lần retry fail trước đó, vì append nằm ngay trước
    return trong nhánh try — chỉ chạy tới được khi request thành công).
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.embeddings.create(
                model=SETTINGS.embedding_model,
                input=texts,
                extra_headers={
                    "HTTP-Referer": SETTINGS.app_referer,
                    "X-Title": SETTINGS.app_title,
                },
            )
            if usage_accumulator is not None and response.usage is not None:
                usage_accumulator.append(response.usage.total_tokens)
            return [item.embedding for item in response.data]
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(f"  Embedding batch failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {last_error}")


def embed_texts(
    texts: list[str], batch_size: int = BATCH_SIZE, usage_accumulator: list[int] | None = None
) -> list[list[float]]:
    """
    Embed danh sách text, tự động chia batch để tránh request quá lớn.
    Trả về list embedding vectors, CÙNG THỨ TỰ với input texts.

    usage_accumulator: nếu truyền vào, token count thật (từ response.usage) của mỗi
    batch được append vào đây — tổng lại (sum) để có total token đã dùng.
    """
    if not texts:
        return []

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} texts)...")
        if usage_accumulator is not None:
            embeddings = _embed_batch(batch, usage_accumulator=usage_accumulator)
        else:
            embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed 1 câu query duy nhất (dùng lúc retrieval, không cần batch)."""
    return _embed_batch([query])[0]
