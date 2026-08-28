"""Client gọi server TEI nội bộ (embedding + reranker), tách biệt hoàn toàn với OpenRouter."""
import requests
from config.settings import SETTINGS

def embed_text_tei(text: str) -> list[float]:
    resp = requests.post(
        f"{SETTINGS.tei_embedding_url}/embed",
        headers={"X-API-Key": SETTINGS.tei_api_key, "Content-Type": "application/json"},
        json={"inputs": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()[0]


def rerank_texts(query: str, texts: list[str]) -> list[dict]:
    """Trả về list [{"index": int, "score": float}], SẮP XẾP theo score giảm dần."""
    resp = requests.post(
        f"{SETTINGS.tei_reranker_url}/rerank",
        headers={"X-API-Key": SETTINGS.tei_api_key, "Content-Type": "application/json"},
        json={"query": query, "texts": texts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()