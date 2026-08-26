"""Extract text từ PDF, giữ nguyên page number cho citation."""
from pathlib import Path

import pymupdf


def extract_pdf_by_page(pdf_path: str) -> list[dict]:
    """
    Đọc PDF, trả về list mỗi phần tử là 1 page.

    Returns:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."},
            ...
        ]

    Raises:
        FileNotFoundError nếu pdf_path không tồn tại.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    doc = pymupdf.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text()
            pages.append({
                "page": page_index + 1,  # 1-indexed cho người đọc
                "text": text.strip(),
            })
    finally:
        doc.close()

    return pages


def extract_pdf_with_metadata(pdf_path: str, source_metadata: dict) -> list[dict]:
    """
    Extract PDF và gắn metadata source cho mỗi page.
    source_metadata là 1 record từ kb/sources.json.

    Returns:
        [
            {
                "source_id": "evn_annual_2024_2025",
                "source": "EVN Annual Report 2024-2025",
                "page": 1,
                "text": "...",
                "organization": "EVN",
                "document_type": "annual_report",
                "year_start": 2024,
                "year_end": 2025,
            },
            ...
        ]
    Chỉ giữ pages có text.strip() không rỗng (skip trang trắng).
    """
    pages = extract_pdf_by_page(pdf_path)
    enriched = []
    for p in pages:
        if not p["text"]:
            continue  # skip blank pages
        enriched.append({
            "source_id": source_metadata["id"],
            "source": source_metadata["title"],
            "page": p["page"],
            "text": p["text"],
            "organization": source_metadata["organization"],
            "document_type": source_metadata["document_type"],
            "year_start": source_metadata["year_start"],
            "year_end": source_metadata["year_end"],
        })
    return enriched
