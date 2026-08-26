"""Retrieval logic: query -> embedding -> Chroma -> formatted results."""
from rag.embeddings import embed_query
from rag.vector_store import query_collection
from rag.citations import format_search_result
from tools.tool_results import success, failure
from config.settings import SETTINGS


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
    for doc, meta, dist in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append(format_search_result(
            text=doc,
            source=meta["source"],
            page=meta["page"],
            distance=dist,
        ))
    return results


def search_knowledge_base(query: str, top_k: int | None = None, source: str | None = None) -> dict:
    """
    Tool-facing function — đây là function mà agent sẽ gọi ở M6/M7.
    Trả về theo format chuẩn ok/data/error đã dùng ở tools/tool_results.py.
    """
    try:
        where = None
        if source:
            where = {"source": source}

        results = retrieve(query, top_k=top_k, where=where)

        if not results:
            return failure(
                "No relevant information found in the knowledge base for this query."
            )

        return success({
            "query": query,
            "results": results,
            "result_count": len(results),
        })
    except Exception as e:
        return failure(f"Knowledge base search failed: {e}")
