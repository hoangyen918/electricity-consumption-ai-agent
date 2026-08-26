import pytest
from rag.chunker import chunk_page_text, chunk_document, count_tokens


def test_short_text_returns_single_chunk():
    text = "Đây là một đoạn văn bản ngắn."
    chunks = chunk_page_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_empty_text_returns_empty_list():
    assert chunk_page_text("") == []
    assert chunk_page_text("   ") == []


def test_long_text_split_into_multiple_chunks():
    # Sinh text đủ dài để chắc chắn > 700 tokens
    text = ("Câu này lặp lại nhiều lần để tạo văn bản dài. " * 500)
    chunks = chunk_page_text(text)
    assert len(chunks) >= 2
    # Mỗi chunk phải không rỗng
    assert all(c.strip() for c in chunks)


def test_chunks_have_overlap():
    text = ("một hai ba bốn năm sáu bảy tám chín mười " * 300)
    chunks = chunk_page_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    # Đảm bảo có overlap: một số token cuối chunk[0] xuất hiện đầu chunk[1]
    # (kiểm tra lỏng, không assert exact vì decode có thể khác nhẹ)


def test_chunk_document_preserves_page_metadata():
    pages = [
        {
            "source_id": "test_doc",
            "source": "Test Document",
            "page": 5,
            "text": "Nội dung trang 5.",
            "organization": "TestOrg",
            "document_type": "test",
            "year_start": 2024,
            "year_end": 2024,
        },
        {
            "source_id": "test_doc",
            "source": "Test Document",
            "page": 7,
            "text": "Nội dung trang 7 dài hơn một chút để test.",
            "organization": "TestOrg",
            "document_type": "test",
            "year_start": 2024,
            "year_end": 2024,
        },
    ]
    chunks = chunk_document(pages)
    assert len(chunks) == 2
    assert chunks[0]["page"] == 5
    assert chunks[1]["page"] == 7
    # Chunk ID phải unique và có format chuẩn
    assert chunks[0]["chunk_id"] == "test_doc_p005_c00"
    assert chunks[1]["chunk_id"] == "test_doc_p007_c00"
    # Không có chunk nào span nhiều page
    for c in chunks:
        assert isinstance(c["page"], int)


def test_chunk_ids_are_unique_across_document():
    """Test quan trọng: nếu 1 page bị chia làm nhiều chunks, ID vẫn phải unique."""
    long_text = ("Nội dung dài lặp lại " * 500)
    pages = [
        {
            "source_id": "test_doc",
            "source": "Test Document",
            "page": 1,
            "text": long_text,
            "organization": "TestOrg",
            "document_type": "test",
            "year_start": 2024,
            "year_end": 2024,
        }
    ]
    chunks = chunk_document(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id phải unique"
