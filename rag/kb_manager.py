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
    """
    Liệt kê MỌI source_id THẬT đang tồn tại trong ChromaDB (nguồn sự thật) — không chỉ
    duyệt theo kb/sources.json như trước. Lý do: nếu add_document() thất bại giữa chừng
    (chunks đã upsert vào Chroma nhưng sources.json chưa kịp ghi — xem sự cố thật
    "world_energy_outlook_2018", 944 chunk mồ côi), nguồn đó vẫn PHẢI hiện trên UI để
    người dùng biết và xóa được, thay vì "lạc trôi" vô hình.

    Nguồn có entry trong sources.json -> dùng metadata đẹp (title, document_type...),
    is_orphan=False. Nguồn KHÔNG có entry (mồ côi) -> title tạm dùng chính source_id,
    các field metadata khác None, is_orphan=True để UI cảnh báo rõ ràng.
    """
    known_sources = load_sources()
    known_ids = {s["id"] for s in known_sources}

    collection = get_collection()
    all_metadatas = collection.get(include=["metadatas"])["metadatas"]
    chunk_counts: dict[str, int] = {}
    for meta in all_metadatas:
        sid = meta.get("source_id", "")
        chunk_counts[sid] = chunk_counts.get(sid, 0) + 1

    result = []
    for src in known_sources:
        result.append({**src, "chunk_count": chunk_counts.get(src["id"], 0), "is_orphan": False})

    for source_id in sorted(set(chunk_counts) - known_ids):
        result.append({
            "id": source_id,
            "title": source_id,
            "filename": None,
            "organization": None,
            "document_type": None,
            "year_start": None,
            "year_end": None,
            "language": None,
            "chunk_count": chunk_counts[source_id],
            "is_orphan": True,
        })
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

    try:
        sources = load_sources()
        sources.append(source_metadata)
        save_sources(sources)
    except Exception as e:
        collection = get_collection()
        collection.delete(where={"source_id": source_id})
        raise RuntimeError(
            f"Ingest thất bại khi ghi kb/sources.json, đã rollback "
            f"{len(chunks)} chunks vừa thêm cho '{source_id}': {e}"
        ) from e

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
