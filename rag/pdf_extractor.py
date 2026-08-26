"""Extract text từ PDF, giữ nguyên page number cho citation."""
import re
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


def extract_txt_by_section(txt_path: str) -> list[dict]:
    """
    Đọc file .txt có cấu trúc mục đánh số ở đầu dòng (1. 2. 3. ...),
    trả về list các dict {"page": số mục, "text": nội dung mục đó}.
    Dùng "page" để tái sử dụng đúng field đã có trong toàn bộ pipeline,
    nhưng ý nghĩa ở đây là số mục, không phải trang PDF.
    Phần mở đầu trước mục "1." được gán page=0.
    """
    text = open(txt_path, encoding="utf-8").read()
    pattern = r'\n(\d{1,2})\.\s'
    parts = re.split(pattern, text)

    sections = []
    if parts[0].strip():
        sections.append({"page": 0, "text": parts[0].strip()})

    for i in range(1, len(parts), 2):
        section_num = int(parts[i])
        section_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if section_text:
            sections.append({"page": section_num, "text": section_text})

    return sections


def extract_txt_with_metadata(txt_path: str, source_metadata: dict) -> list[dict]:
    """Tương tự extract_pdf_with_metadata nhưng cho file .txt theo mục."""
    sections = extract_txt_by_section(txt_path)
    enriched = []
    for s in sections:
        enriched.append({
            "source_id": source_metadata["id"],
            "source": source_metadata["title"],
            "page": s["page"],
            "text": s["text"],
            "organization": source_metadata["organization"],
            "document_type": source_metadata["document_type"],
            "year_start": source_metadata["year_start"],
            "year_end": source_metadata["year_end"],
        })
    return enriched
