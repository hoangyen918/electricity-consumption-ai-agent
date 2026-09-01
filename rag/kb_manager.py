"""Quản lý danh sách nguồn KB — thêm/liệt kê/xóa. Dùng chung cho UI và script."""
import json
import re
import time
from pathlib import Path

from config.settings import SETTINGS
from rag.extractors import extract_document
from rag.chunker import chunk_document
from rag.embeddings import embed_texts
from rag.vector_store import upsert_chunks, get_collection
from services.cost_tracker import calculate_embedding_cost

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
    start_time = time.perf_counter()

    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Định dạng '{ext}' không được hỗ trợ. Chỉ hỗ trợ: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    source_id = _slugify(Path(original_filename).stem)
    existing_sources = load_sources()

    # (a) source_id trùng chính xác — cùng filename từng upload trước đó (slugify ra cùng id).
    duplicate = next((s for s in existing_sources if s["id"] == source_id), None)
    # (b) original_filename khớp field "filename" của 1 entry có sẵn — bắt trường hợp tài liệu
    # đã được curate sẵn (ví dụ qua scripts/ingest_kb.py với id đặt tay như "iea_weo_2021",
    # filename gốc "WorldEnergyOutlook2021.pdf") nhưng user upload lại đúng file đó qua UI,
    # sinh ra source_id khác (do slugify) — không bị (a) bắt vì id khác nhau.
    if duplicate is None:
        duplicate = next((s for s in existing_sources if s["filename"] == original_filename), None)

    if duplicate is not None:
        raise ValueError(
            f"Tài liệu '{original_filename}' đã tồn tại trong kho (source_id: {duplicate['id']}). "
            "Vui lòng xóa bản cũ trước khi nạp lại."
        )

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
    usage_accumulator: list[int] = []
    embeddings = embed_texts(texts, usage_accumulator=usage_accumulator)
    upsert_chunks(chunks, embeddings)

    sources = load_sources()
    sources.append(source_metadata)
    save_sources(sources)

    duration_seconds = time.perf_counter() - start_time
    token_count = sum(usage_accumulator)
    cost_usd = calculate_embedding_cost(SETTINGS.embedding_model, token_count)

    print(
        f"[INGEST] file={original_filename} chunks={len(chunks)} tokens={token_count} "
        f"cost=${cost_usd:.6f} duration={duration_seconds:.2f}s",
        flush=True,
    )

    return {
        "source_id": source_id,
        "chunk_count": len(chunks),
        "token_count": token_count,
        "cost_usd": cost_usd,
        "duration_seconds": duration_seconds,
    }


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
