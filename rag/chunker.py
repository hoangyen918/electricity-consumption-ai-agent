"""Chia page-level text thành token-bounded chunks, giữ nguyên metadata."""
import tiktoken

CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100

# Dùng encoding cl100k_base (chuẩn cho text-embedding-3-small và các model OpenAI hiện tại)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def chunk_page_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
                    overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Chia 1 đoạn text thành các chunk có size xấp xỉ chunk_size tokens,
    với overlap tokens giữa các chunk liên tiếp.
    Trả list các chuỗi text, KHÔNG bao giờ trả chunk rỗng.
    Nếu text ngắn hơn chunk_size thì trả list 1 phần tử là chính text đó.
    """
    if not text.strip():
        return []

    tokens = _ENCODER.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _ENCODER.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += step
    return chunks


def chunk_document(pages: list[dict]) -> list[dict]:
    """
    Nhận output của extract_pdf_with_metadata(), trả list chunks.
    Mỗi chunk chỉ thuộc 1 page (không span nhiều page).

    Returns:
        [
            {
                "chunk_id": "evn_annual_2024_2025_p15_c00",
                "source_id": ...,
                "source": ...,
                "page": 15,
                "chunk_index_in_page": 0,
                "text": "...",
                "token_count": 687,
                "organization": ...,
                "document_type": ...,
                "year_start": ...,
                "year_end": ...,
            },
            ...
        ]
    """
    all_chunks = []
    for page_record in pages:
        page_chunks = chunk_page_text(page_record["text"])
        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{page_record['source_id']}_p{page_record['page']:03d}_c{chunk_idx:02d}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "source_id": page_record["source_id"],
                "source": page_record["source"],
                "page": page_record["page"],
                "chunk_index_in_page": chunk_idx,
                "text": chunk_text,
                "token_count": count_tokens(chunk_text),
                "organization": page_record["organization"],
                "document_type": page_record["document_type"],
                "year_start": page_record["year_start"],
                "year_end": page_record["year_end"],
            })
    return all_chunks
