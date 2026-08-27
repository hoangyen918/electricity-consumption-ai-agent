from rag.extractors import extract_md_with_metadata, extract_document

SAMPLE_MD = """# Tiêu đề tài liệu

Đoạn mở đầu.

## Phần một

Nội dung phần một.

## Phần hai

Nội dung phần hai.
"""

def test_extract_md_splits_by_h2_heading(tmp_path):
    md_path = tmp_path / "test.md"
    md_path.write_text(SAMPLE_MD, encoding="utf-8")

    metadata = {
        "id": "test_doc", "title": "Test Doc", "organization": "Test",
        "document_type": "markdown_doc", "year_start": 2024, "year_end": 2024,
    }
    results = extract_md_with_metadata(str(md_path), metadata)

    assert len(results) == 3  # phần mở đầu (page 0) + 2 heading
    assert results[0]["page"] == 0
    assert results[1]["page"] == 1
    assert "Phần một" in results[1]["text"]
    assert results[2]["page"] == 2
    assert "Phần hai" in results[2]["text"]


def test_extract_document_rejects_unsupported_extension(tmp_path):
    fake_path = tmp_path / "test.json"
    fake_path.write_text("{}", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="Không hỗ trợ"):
        extract_document(str(fake_path), {})
