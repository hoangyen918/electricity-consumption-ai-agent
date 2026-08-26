"""Format citation string từ metadata retrieval — không để LLM tự bịa citation."""

def format_citation(source: str, page: int) -> str:
    """Trả về dạng '[EVN Annual Report 2024-2025, p.9]'"""
    return f"[{source}, p.{page}]"


def format_search_result(text: str, source: str, page: int, distance: float) -> dict:
    """Chuẩn hóa 1 kết quả retrieval thành dict nhất quán."""
    return {
        "text": text,
        "source": source,
        "page": page,
        "citation": format_citation(source, page),
        "distance": round(distance, 4),
    }
