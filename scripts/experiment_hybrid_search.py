"""
THỬ NGHIỆM hybrid search (BM25 + vector) so với pure vector hiện tại.
KHÔNG tự động áp dụng vào production — chỉ đo và báo cáo, quyết định giữ/bỏ dựa trên số liệu.
Chạy: python scripts/experiment_hybrid_search.py
"""
import sys
import string
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from rank_bm25 import BM25Okapi

from rag.vector_store import get_collection
from rag.embeddings import embed_query
from rag.retriever import retrieve as pure_vector_retrieve


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def build_bm25_index():
    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]
    tokenized = [simple_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    return bm25, documents, metadatas, ids


def hybrid_search_rrf(query: str, bm25, documents, metadatas, ids, top_k: int = 10, k_constant: int = 60):
    """Reciprocal Rank Fusion — kết hợp thứ hạng, không kết hợp điểm số trực tiếp."""
    # Vector ranking
    query_embedding = embed_query(query)
    collection = get_collection()
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=len(ids))
    vector_ranked_ids = vector_results["ids"][0]

    # BM25 ranking
    tokenized_query = simple_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
    bm25_ranked_ids = [ids[i] for i in bm25_ranked_indices]

    # RRF fusion
    rrf_scores = {}
    for rank, doc_id in enumerate(vector_ranked_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_constant + rank + 1)
    for rank, doc_id in enumerate(bm25_ranked_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_constant + rank + 1)

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    id_to_idx = {doc_id: i for i, doc_id in enumerate(ids)}
    results = []
    for doc_id in sorted_ids:
        idx = id_to_idx[doc_id]
        results.append({
            "source": metadatas[idx]["source"],
            "page": metadatas[idx]["page"],
            "rrf_score": rrf_scores[doc_id],
        })
    return results


def main():
    print("Building BM25 index từ toàn bộ collection hiện tại...")
    bm25, documents, metadatas, ids = build_bm25_index()
    print(f"  Đã index {len(documents)} chunks\n")

    # Dùng đúng các case đã biết ground truth để so sánh công bằng
    test_cases = [
        ("Tỷ lệ hộ dân nông thôn có điện là bao nhiêu?", "2024-2025", 9),  # case xfail đã biết
        ("Tổng sản lượng điện thương phẩm năm 2023 là bao nhiêu?", "2022-2023", 10),
        ("Tỷ lệ tổn thất điện năng của EVN là bao nhiêu?", "2022-2023", 10),
    ]

    print("=" * 70)
    print("SO SÁNH: Pure Vector (hiện tại) vs Hybrid RRF (thử nghiệm)")
    print("=" * 70)

    for question, expected_source_fragment, expected_page in test_cases:
        print(f"\nCâu hỏi: {question}")
        print(f"Kỳ vọng: nguồn chứa '{expected_source_fragment}', trang {expected_page}")

        # Pure vector (baseline hiện tại)
        vector_results = pure_vector_retrieve(question, top_k=10)
        vector_rank = next(
            (i + 1 for i, r in enumerate(vector_results)
             if expected_source_fragment in r["source"] and r["page"] == expected_page),
            None
        )
        print(f"  Pure Vector: hạng #{vector_rank if vector_rank else 'KHÔNG lọt top-10'}")

        # Hybrid RRF
        hybrid_results = hybrid_search_rrf(question, bm25, documents, metadatas, ids, top_k=10)
        hybrid_rank = next(
            (i + 1 for i, r in enumerate(hybrid_results)
             if expected_source_fragment in r["source"] and r["page"] == expected_page),
            None
        )
        print(f"  Hybrid RRF:  hạng #{hybrid_rank if hybrid_rank else 'KHÔNG lọt top-10'}")

        if vector_rank and hybrid_rank:
            if hybrid_rank < vector_rank:
                print(f"  => Hybrid CẢI THIỆN (hạng {vector_rank} -> {hybrid_rank})")
            elif hybrid_rank > vector_rank:
                print(f"  => Hybrid LÀM TỆ HƠN (hạng {vector_rank} -> {hybrid_rank})")
            else:
                print(f"  => Không đổi")

    print("\n" + "=" * 70)
    print("KẾT LUẬN: đọc kết quả trên, chỉ áp dụng hybrid vào production nếu")
    print("thấy CẢI THIỆN RÕ RỆT trên case xfail, KHÔNG làm tệ hơn 2 case còn lại.")
    print("=" * 70)


if __name__ == "__main__":
    main()
