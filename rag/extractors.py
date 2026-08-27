"""
Registry các hàm extract theo định dạng file KB.
Mỗi hàm nhận (file_path, source_metadata), trả về list[dict] dạng:
[{"source_id", "source", "page", "text", "organization", "document_type", "year_start", "year_end"}]
"page" mang ý nghĩa khác nhau tùy định dạng: số trang PDF thật, số mục văn bản luật (.txt),
hoặc số thứ tự heading (.md) — miễn có tính thứ tự và dùng nhất quán cho citation.
"""
import re
from pathlib import Path

import pymupdf


def _wrap(source_metadata: dict, page: int, text: str) -> dict:
    return {
        "source_id": source_metadata["id"],
        "source": source_metadata["title"],
        "page": page,
        "text": text,
        "organization": source_metadata["organization"],
        "document_type": source_metadata["document_type"],
        "year_start": source_metadata["year_start"],
        "year_end": source_metadata["year_end"],
    }


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


def extract_pdf_with_metadata(file_path: str, source_metadata: dict) -> list[dict]:
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
    pages = extract_pdf_by_page(file_path)
    return [_wrap(source_metadata, p["page"], p["text"]) for p in pages if p["text"]]


def extract_txt_by_section(txt_path: str) -> list[dict]:
    """
    Đọc file .txt có cấu trúc mục đánh số ở đầu dòng (1. 2. 3. ...),
    trả về list các dict {"page": số mục, "text": nội dung mục đó}.
    Dùng "page" để tái sử dụng đúng field đã có trong toàn bộ pipeline,
    nhưng ý nghĩa ở đây là số mục, không phải trang PDF.
    Phần mở đầu trước mục "1." được gán page=0.
    """
    text = Path(txt_path).read_text(encoding="utf-8")
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


def extract_txt_with_metadata(file_path: str, source_metadata: dict) -> list[dict]:
    """Tương tự extract_pdf_with_metadata nhưng cho file .txt theo mục."""
    sections = extract_txt_by_section(file_path)
    return [_wrap(source_metadata, s["page"], s["text"]) for s in sections]


def extract_md_with_metadata(file_path: str, source_metadata: dict) -> list[dict]:
    """
    Chia Markdown theo heading cấp 2 (## Tiêu đề). Mỗi heading là 1 "page"
    (số thứ tự heading, không phải số trang thật). Phần trước heading đầu
    tiên (nếu có) được gán page=0.
    """
    text = Path(file_path).read_text(encoding="utf-8")
    sections = re.split(r'\n##\s+', text)

    results = []
    if sections[0].strip():
        results.append(_wrap(source_metadata, 0, sections[0].strip()))
    for idx, section in enumerate(sections[1:], start=1):
        if section.strip():
            results.append(_wrap(source_metadata, idx, section.strip()))
    return results


EXTRACTOR_REGISTRY = {
    ".pdf": extract_pdf_with_metadata,
    ".txt": extract_txt_with_metadata,
    ".md": extract_md_with_metadata,
}


def extract_document(file_path: str, source_metadata: dict) -> list[dict]:
    """Điểm vào duy nhất — tự chọn đúng extractor theo phần mở rộng file."""
    ext = Path(file_path).suffix.lower()
    if ext not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Không hỗ trợ định dạng '{ext}'. Định dạng hỗ trợ: {list(EXTRACTOR_REGISTRY.keys())}"
        )
    return EXTRACTOR_REGISTRY[ext](file_path, source_metadata)
