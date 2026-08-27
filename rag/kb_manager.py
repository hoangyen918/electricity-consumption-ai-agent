"""Quản lý danh sách nguồn KB — thêm/liệt kê/xóa. Dùng chung cho UI và script."""
import json
import re
from pathlib import Path

from rag.extractors import extract_document
from rag.chunker import chunk_document
from rag.embeddings import embed_texts
from rag.vector_store import upsert_chunks, get_collection

SOURCES_PATH = Path("kb/sources.json")
RAW_DIR = Path("kb/raw")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_sources() -> list[dict]:
    if not SOURCES_PATH.exists():
        return []
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def save_sources(sources: list[dict]) -> None:
    SOURCES_PATH.write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_sources_with_chunk_counts() -> list[dict]:
    """Danh sách nguồn kèm số chunk THẬT đang có trong Chroma (không lấy từ sources.json,
    vì sources.json chỉ ghi metadata, số chunk thật phải hỏi Chroma để tránh lệch)."""
    sources = load_sources()
    collection = get_collection()
    result = []
    for src in sources:
        matches = collection.get(where={"source_id": src["id"]})
        chunk_count = len(matches["ids"]) if matches and matches.get("ids") else 0
        result.append({**src, "chunk_count": chunk_count})
    return result


def add_document(
    file_bytes: bytes,
    original_filename: str,
    title: str,
    organization: str = "User Upload",
    document_type: str = "user_document",
    year_start: str = "",
    year_end: str = "",
    language: str = "",
) -> dict:
    """
    Thêm 1 tài liệu mới: lưu file vào kb/raw/, extract, chunk, embed, upsert vào Chroma,
    cập nhật kb/sources.json. year_start/year_end/language dùng chuỗi rỗng mặc định, KHÔNG
    dùng None vì ChromaDB không chấp nhận None trong metadata (sẽ crash lúc upsert).
    """
    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Định dạng '{ext}' không được hỗ trợ. Chỉ hỗ trợ: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    source_id = _slugify(Path(original_filename).stem)
    existing_ids = {s["id"] for s in load_sources()}
    if source_id in existing_ids:
        suffix = 2
        while f"{source_id}_{suffix}" in existing_ids:
            suffix += 1
        source_id = f"{source_id}_{suffix}"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved_filename = f"{source_id}{ext}"
    saved_path = RAW_DIR / saved_filename
    saved_path.write_bytes(file_bytes)

    source_metadata = {
        "id": source_id,
        "title": title,
        "filename": saved_filename,
        "organization": organization,
        "document_type": document_type,
        "year_start": year_start,
        "year_end": year_end,
        "language": language,
    }

    pages = extract_document(str(saved_path), source_metadata)
    chunks = chunk_document(pages)
    if not chunks:
        saved_path.unlink(missing_ok=True)
        raise ValueError("Không trích xuất được nội dung nào từ file này (file rỗng hoặc lỗi định dạng).")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert_chunks(chunks, embeddings)

    sources = load_sources()
    sources.append(source_metadata)
    save_sources(sources)

    return {"source_id": source_id, "chunk_count": len(chunks)}


def delete_document(source_id: str) -> int:
    """Xóa 1 nguồn khỏi Chroma + kb/sources.json + file gốc trong kb/raw/. Trả số chunk đã xóa."""
    collection = get_collection()
    matches = collection.get(where={"source_id": source_id})
    deleted_count = len(matches["ids"]) if matches and matches.get("ids") else 0

    if deleted_count > 0:
        collection.delete(where={"source_id": source_id})

    sources = load_sources()
    target = next((s for s in sources if s["id"] == source_id), None)
    sources = [s for s in sources if s["id"] != source_id]
    save_sources(sources)

    if target:
        (RAW_DIR / target["filename"]).unlink(missing_ok=True)

    return deleted_count
