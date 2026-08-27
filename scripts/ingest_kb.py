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
from rag.vector_store import upsert_chunks, count_documents


def main():
    sources_path = Path("kb/sources.json")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))

    all_chunks = []

    print("=" * 60)
    print("STEP 1: Extract + chunk all documents")
    print("=" * 60)
    for src in sources:
        doc_path = Path("kb/raw") / src["filename"]
        if not doc_path.exists():
            print(f"⚠️  SKIP — file missing: {doc_path}")
            continue

        try:
            pages = extract_document(str(doc_path), src)
        except ValueError as e:
            print(f"⚠️  SKIP — {e}")
            continue

        chunks = chunk_document(pages)
        all_chunks.extend(chunks)
        print(f"  {src['title']}: {len(pages)} pages -> {len(chunks)} chunks")

    if not all_chunks:
        print("ERROR: No chunks produced. Check kb/raw/ has the PDF files.")
        return

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    print("\n" + "=" * 60)
    print("STEP 2: Generate embeddings via OpenRouter")
    print("=" * 60)
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)
    print(f"  Generated {len(embeddings)} embeddings")

    print("\n" + "=" * 60)
    print("STEP 3: Upsert into ChromaDB")
    print("=" * 60)
    upsert_chunks(all_chunks, embeddings)

    total_in_db = count_documents()
    print(f"  Collection now has {total_in_db} documents total")

    print("\n" + "=" * 60)
    print(f"DONE. Ingested {len(all_chunks)} chunks from {len(sources)} sources.")
    print(f"Collection total: {total_in_db} documents")
    print("=" * 60)


if __name__ == "__main__":
    main()
