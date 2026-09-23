"""
Chạy toàn bộ pipeline: PDF -> extract -> chunk -> embed -> ChromaDB.
An toàn để chạy lại nhiều lần (dùng upsert, không nhân đôi dữ liệu).
Chạy: python scripts/ingest_kb.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag.extractors import extract_document
from rag.chunker import chunk_document
from rag.embeddings import embed_texts
from rag.vector_store import upsert_chunks, count_documents, get_collection


def main():
    sources_path = Path("kb/sources.json")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))

    # Embed + upsert NGAY sau khi chunk từng source (không dồn hết vào 1 list rồi embed
    # 1 lần ở cuối) — vì upsert_chunks() đã idempotent (ghi đè theo chunk_id), nếu
    # server embedding local hiccup/treo giữa đường (đã xảy ra thật 2026-09-22, server
    # Ollama treo ở batch 31/63, mất trắng ~3000 embedding đã tính vì chưa kịp ghi),
    # các source ĐÃ upsert xong vẫn được giữ lại, rerun chỉ cần làm lại từ source lỗi.
    # Skip source đã có chunk trong Chroma (source_id metadata) để rerun sau lỗi không
    # phải embed lại từ đầu — muốn ép re-embed 1 source cụ thể thì xóa nó qua
    # rag/kb_manager.py:delete_document() trước rồi chạy lại.
    existing_source_ids = {
        m.get("source_id", "") for m in get_collection().get(include=["metadatas"])["metadatas"]
    }

    total_chunks = 0
    failed_sources = []
    print("=" * 60)
    print("Extract -> chunk -> embed -> upsert, PER SOURCE")
    print("=" * 60)
    for src in sources:
        if src["id"] in existing_source_ids:
            print(f"  SKIP (already in Chroma) {src['title']}")
            continue

        doc_path = Path("kb/raw") / src["filename"]
        if not doc_path.exists():
            print(f"⚠️  SKIP — file missing: {doc_path}")
            continue

        try:
            pages = extract_document(str(doc_path), src)
            chunks = chunk_document(pages)
            if not chunks:
                print(f"⚠️  SKIP — {src['title']}: no chunks produced")
                continue

            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts)
            upsert_chunks(chunks, embeddings)
            total_chunks += len(chunks)
            print(f"  OK  {src['title']}: {len(pages)} pages -> {len(chunks)} chunks embedded+upserted")
        except ValueError as e:
            print(f"⚠️  SKIP — {e}")
        except Exception as e:
            print(f"❌ FAILED — {src['title']}: {e}")
            failed_sources.append(src["title"])

    total_in_db = count_documents()
    print("\n" + "=" * 60)
    print(f"DONE. Ingested {total_chunks} chunks from {len(sources) - len(failed_sources)}/{len(sources)} sources.")
    if failed_sources:
        print(f"FAILED sources (rerun this script to retry — sources already in Chroma will be skipped): {failed_sources}")
    print(f"Collection total: {total_in_db} documents")
    print("=" * 60)


if __name__ == "__main__":
    main()
