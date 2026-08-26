"""Format citation string từ metadata retrieval — không để LLM tự bịa citation."""

def format_citation(source: str, page: int, document_type: str = "annual_report") -> str:
    """Trả về dạng '[EVN Annual Report 2024-2025, p.9]' hoặc '[Chỉ thị 20/CT-TTg, Mục 5]'."""
    if document_type == "policy_directive":
        if page == 0:
            return f"[{source}]"
        return f"[{source}, Mục {page}]"
    return f"[{source}, p.{page}]"


def format_search_result(text: str, source: str, page: int, distance: float, document_type: str = "annual_report") -> dict:
    """Chuẩn hóa 1 kết quả retrieval thành dict nhất quán."""
    return {
        "text": text,
        "source": source,
        "page": page,
        "citation": format_citation(source, page, document_type),
        "distance": round(distance, 4),
    }
