"""
Đọc 2 PDF EVN, chia chunks, in báo cáo. KHÔNG gọi API, KHÔNG ghi vào Chroma.
Chạy: python scripts/inspect_pdfs.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag.pdf_extractor import extract_pdf_with_metadata
from rag.chunker import chunk_document


def main():
    sources_path = Path("kb/sources.json")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))

    total_pages = 0
    total_chunks = 0
    total_tokens = 0

    for src in sources:
        pdf_path = Path("kb/raw") / src["filename"]
        print(f"\n{'='*60}")
        print(f"Source: {src['title']}")
        print(f"File: {pdf_path}")

        if not pdf_path.exists():
            print(f"  ⚠️  FILE MISSING — skip. Tải PDF vào kb/raw/{src['filename']} trước.")
            continue

        pages = extract_pdf_with_metadata(str(pdf_path), src)
        chunks = chunk_document(pages)

        doc_tokens = sum(c["token_count"] for c in chunks)
        total_pages += len(pages)
        total_chunks += len(chunks)
        total_tokens += doc_tokens

        print(f"  Pages with text: {len(pages)}")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Total tokens: {doc_tokens:,}")
        print(f"  Avg tokens/chunk: {doc_tokens / len(chunks) if chunks else 0:.1f}")

        # In sample chunk đầu và giữa để user mắt thấy đọc có đúng không
        if chunks:
            print(f"\n  --- Sample chunk (first) ---")
            print(f"  chunk_id: {chunks[0]['chunk_id']}")
            print(f"  page: {chunks[0]['page']}")
            preview = chunks[0]['text'][:300].replace("\n", " ")
            print(f"  text preview: {preview}...")

            if len(chunks) > 10:
                mid = len(chunks) // 2
                print(f"\n  --- Sample chunk (middle #{mid}) ---")
                print(f"  chunk_id: {chunks[mid]['chunk_id']}")
                print(f"  page: {chunks[mid]['page']}")
                preview = chunks[mid]['text'][:300].replace("\n", " ")
                print(f"  text preview: {preview}...")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_pages} pages, {total_chunks} chunks, {total_tokens:,} tokens total")
    print(f"Ước lượng cost embedding (text-embedding-3-small, $0.02/1M):")
    est_cost = total_tokens * 0.02 / 1_000_000
    print(f"  ~${est_cost:.6f} (một lần duy nhất khi ingest ở M4)")


if __name__ == "__main__":
    main()
