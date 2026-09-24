"""Retrieval logic: query -> embedding -> Chroma -> formatted results."""
from rag.embeddings import embed_query
from rag.vector_store import query_collection
from rag.citations import format_search_result
from rag.chunker import count_tokens
from tools.tool_results import success, failure
from config.settings import SETTINGS

# Đo lại 2026-09-24 sau khi đổi embedding model sang bge-m3 (thang đo distance khác
# hẳn model cũ, giá trị 1.3 cũ KHÔNG áp dụng được — xem DATA_SOURCES.md). Known-good
# distance thật đo trên VERIFIED_CASES: 0.8327 - 1.0020 (cao nhất). Nonsense-query floor
# đo được: 1.175 - 1.211. Chọn 1.1 = cao nhất known-good + margin ~0.1, vẫn thấp hơn sàn
# nonsense ~0.075 — cùng phương pháp đã dùng cho giá trị 1.3 cũ, xem DATA_SOURCES.md.
SCORE_THRESHOLD = 1.1


def retrieve(query: str, top_k: int | None = None, where: dict | None = None) -> list[dict]:
    """
    Retrieval nội bộ — trả list kết quả đã format, KHÔNG wrap ok/data/error
    (dùng cho debug script, test, và bên trong search_knowledge_base).
    """
    if top_k is None:
        top_k = SETTINGS.rag_top_k

    if not query or not query.strip():
        return []

    query_embedding = embed_query(query)
    raw = query_collection(query_embedding, top_k=top_k, where=where)

    if not raw["documents"] or not raw["documents"][0]:
        return []

    results = []
    for cid, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append(format_search_result(
            text=doc,
            source=meta["source"],
            page=meta["page"],
            distance=dist,
            document_type=meta.get("document_type", "annual_report"),
            chunk_id=cid,
        ))
    return results


def search_knowledge_base(
    query: str,
    top_k: int | None = None,
    source: str | None = None,
    language: str | None = None,
    document_type: str | None = None,
) -> dict:
    """
    Tool-facing function — đây là function mà agent sẽ gọi ở M6/M7.
    Trả về theo format chuẩn ok/data/error đã dùng ở tools/tool_results.py.
    """
    try:
        conditions = []
        if source:
            conditions.append({"source": source})
        if language:
            conditions.append({"language": language})
        if document_type:
            conditions.append({"document_type": document_type})

        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        results = retrieve(query, top_k=top_k, where=where)
        filtered_results = [r for r in results if r["distance"] <= SCORE_THRESHOLD]

        if not filtered_results:
            return failure(
                "No relevant information found in the knowledge base for this query "
                "(closest matches were too dissimilar to be considered reliable)."
            )

        return success({
            "query": query,
            "results": filtered_results,
            "result_count": len(filtered_results),
            "embedding_tokens": count_tokens(query),
        })
    except Exception as e:
        return failure(f"Knowledge base search failed: {e}")
