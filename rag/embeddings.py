"""Gọi embedding model qua TEI self-hosted (OpenAI-compatible /v1/embeddings), có batching và retry.

Đổi 2026-09-23 từ Ollama-native /api/embed sang TEI tự host trong docker-compose.yml
(ghcr.io/huggingface/text-embeddings-inference) — server LAN Ollama treo giữa ingest,
server TEI công ty (74bt.ats.com.vn) không phản hồi liên tục. Response shape lại giống
OpenAI: {"data": [{"embedding": [...], "index": N}, ...], "usage": {"total_tokens": N}}.
"""
import time
import requests
from config.settings import SETTINGS

# Giới hạn CỨNG của backend TEI đo được thật (2026-09-23), không phải số tùy chọn:
# batch=8 -> 200 OK (547ms); batch=32 -> 429 "Model is overloaded" (dù đã nâng
# --max-concurrent-requests 4->16 trong docker-compose.yml, không phải do CPU/RAM vẫn
# ở mức idle); batch=100 -> 422 (vượt --max-client-batch-size=32). Log warm-up server
# ghi rõ "Backend does not support a batch size > 8, forcing max_batch_requests=8".
# KHÔNG tăng số này mà không đo lại bằng docker stats trong lúc gửi batch thật.
BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # nhân đôi mỗi lần retry: 2s, 4s (chỉ chờ giữa các lần thử, không chờ sau lần cuối)
TIMEOUT_SECONDS = 120

# 429 (rate limit / "Model is overloaded", đã đo thật) và 5xx là lỗi TẠM THỜI -> retry
# với backoff hợp lý. 4xx khác (400 malformed, 422 batch vượt max-client-batch-size) là
# lỗi CỐ ĐỊNH — request y hệt sẽ luôn fail y hệt, retry 3 lần chỉ tốn ~6s vô ích trước
# khi vẫn báo lỗi, nên fail ngay thay vì retry.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is None:
        return True  # lỗi kết nối/timeout, không có response — tạm thời, nên retry
    return response.status_code in _RETRYABLE_STATUS_CODES


def _embed_batch(texts: list[str], usage_accumulator: list[int] | None = None) -> list[list[float]]:
    """Gọi API cho 1 batch, có retry với exponential backoff cho lỗi TẠM THỜI (429/5xx/
    kết nối/timeout). Lỗi CỐ ĐỊNH (400/422/...) raise ngay, không retry — xem
    _is_retryable().

    usage_accumulator: nếu truyền vào, token count (usage.total_tokens) của LẦN GỌI
    THÀNH CÔNG được append vào đây (không cộng dồn các lần retry fail trước đó, vì
    append nằm ngay trước return trong nhánh try — chỉ chạy tới được khi request
    thành công).
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                SETTINGS.embedding_base_url,
                json={"model": SETTINGS.embedding_model, "input": texts},
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            if usage_accumulator is not None and "usage" in payload:
                usage_accumulator.append(payload["usage"]["total_tokens"])
            return [item["embedding"] for item in payload["data"]]
        except Exception as e:
            if not _is_retryable(e):
                raise RuntimeError(f"Embedding request bị từ chối (lỗi cố định, không retry): {e}") from e
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

    usage_accumulator: nếu truyền vào, token count thật (usage.total_tokens) của mỗi
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
    """Embed 1 câu query duy nhất (dùng lúc retrieval, không cần batch).

    Cùng retry/backoff với embed_texts() — quan trọng cho đường dẫn RAG search khi app
    chạy thật: nếu embed_query() đụng 429 (backend đang xử lý batch ingest khác cùng
    lúc), phải retry cho user, không được để lỗi thẳng ra ngay lần 429 đầu tiên.
    """
    return _embed_batch([query])[0]
