"""
Debug tool: xem tổng quan collection ChromaDB hiện tại, thử 1 query mẫu.
Chạy: python scripts/inspect_chroma.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag.vector_store import count_documents, query_collection, get_collection
from rag.embeddings import embed_query


def main():
    total = count_documents()
    print(f"Collection '{get_collection().name}' has {total} documents.\n")

    if total == 0:
        print("Collection is empty. Run: python scripts/ingest_kb.py")
        return

    # Thử 1 query mẫu liên quan tới nội dung EVN
    sample_query = "Ngành nào chiếm tỷ trọng điện thương phẩm lớn nhất?"
    print(f"Sample query: {sample_query}\n")

    query_embedding = embed_query(sample_query)
    results = query_collection(query_embedding, top_k=3)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        print(f"--- Rank {rank} (distance={dist:.4f}) ---")
        print(f"Source: {meta['source']}, page {meta['page']}")
        preview = doc[:200].replace("\n", " ")
        print(f"Text: {preview}...\n")


if __name__ == "__main__":
    main()
