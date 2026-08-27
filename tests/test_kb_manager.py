"""Test add/list/delete document — dùng tmp_path để không đụng KB thật."""
import pytest
import rag.kb_manager as kb_manager


@pytest.fixture(autouse=True)
def isolate_kb(tmp_path, monkeypatch):
    """Trỏ SOURCES_PATH và RAW_DIR sang thư mục tạm, không đụng kb/ thật."""
    fake_sources = tmp_path / "sources.json"
    fake_sources.write_text("[]", encoding="utf-8")
    fake_raw = tmp_path / "raw"
    fake_raw.mkdir()
    monkeypatch.setattr(kb_manager, "SOURCES_PATH", fake_sources)
    monkeypatch.setattr(kb_manager, "RAW_DIR", fake_raw)


def test_add_document_txt_creates_chunks():
    content = b"1. Noi dung muc mot ve tiet kiem dien.\n2. Noi dung muc hai."
    result = kb_manager.add_document(content, "test_doc.txt", "Test Document")
    try:
        assert result["chunk_count"] > 0

        sources = kb_manager.load_sources()
        assert len(sources) == 1
        assert sources[0]["title"] == "Test Document"
        assert sources[0]["year_start"] == ""  # không phải None
    finally:
        kb_manager.delete_document(result["source_id"])


def test_add_document_rejects_unsupported_format():
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        kb_manager.add_document(b"{}", "test.json", "Test")


def test_add_document_duplicate_filename_gets_unique_id():
    content = b"1. Noi dung test."
    r1 = kb_manager.add_document(content, "duplicate.txt", "Doc 1")
    try:
        r2 = kb_manager.add_document(content, "duplicate.txt", "Doc 2")
        try:
            assert r1["source_id"] != r2["source_id"]

            sources = kb_manager.load_sources()
            assert len(sources) == 2
        finally:
            kb_manager.delete_document(r2["source_id"])
    finally:
        kb_manager.delete_document(r1["source_id"])


def test_delete_document_removes_from_sources_json():
    content = b"1. Noi dung se bi xoa."
    result = kb_manager.add_document(content, "to_delete.txt", "To Delete")
    source_id = result["source_id"]

    deleted_count = kb_manager.delete_document(source_id)
    assert deleted_count > 0

    sources = kb_manager.load_sources()
    assert all(s["id"] != source_id for s in sources)


def test_delete_nonexistent_document_returns_zero():
    deleted_count = kb_manager.delete_document("khong_ton_tai_id")
    assert deleted_count == 0


def test_list_sources_with_chunk_counts_matches_reality():
    content = b"1. Mot doan van ban du dai de tao it nhat mot chunk."
    result = kb_manager.add_document(content, "count_test.txt", "Count Test")
    try:
        listed = kb_manager.list_sources_with_chunk_counts()
        matching = [s for s in listed if s["id"] == result["source_id"]]
        assert len(matching) == 1
        assert matching[0]["chunk_count"] == result["chunk_count"]
    finally:
        kb_manager.delete_document(result["source_id"])
