"""ChromaDB PersistentClient wrapper — lưu/truy vấn vector cục bộ."""
import chromadb
from config.settings import SETTINGS

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=SETTINGS.chroma_path)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(name=SETTINGS.chroma_collection)
    return _collection


UPSERT_BATCH_SIZE = 1000  # ChromaDB có giới hạn max batch size (server đã gặp: 5461) — chia nhỏ để an toàn


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Ghi chunks vào Chroma bằng upsert (không phải add) — chạy lại nhiều lần
    với cùng chunk_id sẽ GHI ĐÈ, không tạo bản sao.

    chunks: list dict có chunk_id, text, và các metadata field khác.
    embeddings: list vector, CÙNG THỨ TỰ với chunks.
    Tự động chia batch để không vượt quá giới hạn batch size của ChromaDB.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

    collection = get_collection()

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_id": c["source_id"],
            "source": c["source"],
            "page": c["page"],
            "organization": c["organization"],
            "document_type": c["document_type"],
            "year_start": c["year_start"],
            "year_end": c["year_end"],
            "language": c.get("language", ""),
        }
        for c in chunks
    ]

    for i in range(0, len(ids), UPSERT_BATCH_SIZE):
        end = i + UPSERT_BATCH_SIZE
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )


def count_documents() -> int:
    return get_collection().count()


def query_collection(query_embedding: list[float], top_k: int = 5, where: dict | None = None) -> dict:
    """Query Chroma bằng embedding đã tính sẵn."""
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def reset_collection() -> None:
    """XÓA TOÀN BỘ collection — chỉ dùng khi debug, không dùng trong ingest bình thường."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(name=SETTINGS.chroma_collection)
    except Exception:
        pass
    _collection = None
