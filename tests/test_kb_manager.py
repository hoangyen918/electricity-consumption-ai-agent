"""Test add/list/delete document — dùng tmp_path để không đụng KB thật."""
from unittest.mock import patch

import pytest
import rag.kb_manager as kb_manager
from config.settings import SETTINGS
from services.cost_tracker import calculate_embedding_cost


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


def test_add_document_duplicate_filename_is_blocked_not_auto_suffixed():
    """
    Hành vi CŨ (auto-suffix source_id trùng thành _2, _3...) đã bị thay bằng chặn +
    raise ValueError — đây chính là fix cho bug orphan chunk thật đã xảy ra (upload
    lại 1 tài liệu đã curate sẵn dưới tên khác sinh source_id khác, xóa nhầm bản cũ
    để lại orphan). Test này thay thế test_add_document_duplicate_filename_gets_unique_id
    cũ (đã xóa) vì test đó khẳng định đúng hành vi giờ bị coi là bug.
    """
    content = b"1. Noi dung test."
    r1 = kb_manager.add_document(content, "duplicate.txt", "Doc 1")
    try:
        with pytest.raises(ValueError, match="duplicate.txt"):
            kb_manager.add_document(content, "duplicate.txt", "Doc 2")

        sources = kb_manager.load_sources()
        assert len(sources) == 1
    finally:
        kb_manager.delete_document(r1["source_id"])


def test_add_document_reupload_same_filename_blocked_with_clear_message():
    """Bước 5b: upload cùng filename lần 2 -> ValueError, message chứa tên file + source_id cũ."""
    content = b"1. Mot doan van ban ngan."
    r1 = kb_manager.add_document(content, "report_v1.txt", "Report V1")
    try:
        with pytest.raises(ValueError) as exc_info:
            kb_manager.add_document(content, "report_v1.txt", "Report V1 (lai)")

        message = str(exc_info.value)
        assert "report_v1.txt" in message
        assert r1["source_id"] in message
    finally:
        kb_manager.delete_document(r1["source_id"])


def test_add_document_different_names_same_slugified_id_blocked():
    """
    Bước 5c: "my-doc.txt" và "my_doc.txt" đều slugify(stem) ra "my_doc" (regex
    [^a-z0-9]+ khớp cả '-' lẫn '_', xác nhận qua _slugify() thật trước khi viết test
    này). Upload file thứ 2 với source_id trùng (dù tên khác) phải bị chặn.
    """
    assert kb_manager._slugify("my-doc") == "my_doc"
    assert kb_manager._slugify("my_doc") == "my_doc"

    content = b"1. Noi dung bat ky."
    r1 = kb_manager.add_document(content, "my-doc.txt", "My Doc Dash")
    try:
        with pytest.raises(ValueError, match="my_doc"):
            kb_manager.add_document(content, "my_doc.txt", "My Doc Underscore")

        sources = kb_manager.load_sources()
        assert len(sources) == 1
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


def test_add_document_reports_embedding_tokens_cost_and_duration():
    """
    Regression test cho observability mới (M-ingest-cost): add_document() phải
    cộng dồn token thật từ response.usage (qua usage_accumulator của embed_texts),
    tính cost bằng đúng calculate_embedding_cost() có sẵn (KHÔNG hard-code công thức
    riêng), và đo duration của toàn bộ khối ingest. Mock _embed_batch để không tốn
    tiền thật lúc chạy pytest, đúng pattern mock đã dùng trong tests/test_embeddings.py.
    """
    content = (
        b"1. Tiet kiem dien la trach nhiem cua moi ho gia dinh.\n"
        b"2. Su dung thiet bi hieu suat cao giup giam hoa don tien dien.\n"
        b"3. Dieu hoa nen dat o muc 26 do C tro len de tiet kiem nang luong.\n"
        b"4. Tat cac thiet bi khong su dung de tranh lang phi dien nang.\n"
        b"5. EVN khuyen cao su dung dien vao gio thap diem de giam tai luoi.\n"
        b"6. Nang luong tai tao dang duoc khuyen khich phat trien tai Viet Nam.\n"
        b"7. Kiem tra dinh ky cac thiet bi dien de dam bao an toan.\n"
        b"8. Su dung den LED thay cho den soi dot truyen thong.\n"
        b"9. Rut phich cam khi khong su dung thiet bi trong thoi gian dai.\n"
        b"10. Theo doi hoa don dien hang thang de kiem soat muc tieu thu.\n"
    )

    def fake_embed_batch(batch, usage_accumulator=None):
        """
        Vector giả PHẢI đủ 1536 chiều (dimension thật của text-embedding-3-small):
        collection Chroma thật ở storage/chroma/ đã cố định dimension này từ các
        tài liệu embed thật trước đó, upsert vector sai dimension sẽ raise
        InvalidArgumentError bất kể nội dung test có mock embedding hay không.
        """
        if usage_accumulator is not None:
            usage_accumulator.append(37 * len(batch))
        return [[0.001] * 1536 for _ in batch]

    with patch("rag.embeddings._embed_batch", side_effect=fake_embed_batch):
        result = kb_manager.add_document(content, "cost_test.txt", "Cost Test")

    try:
        assert result["chunk_count"] > 0
        assert result["token_count"] > 0
        assert result["cost_usd"] > 0
        assert result["duration_seconds"] > 0

        expected_cost = calculate_embedding_cost(SETTINGS.embedding_model, result["token_count"])
        assert result["cost_usd"] == expected_cost
    finally:
        kb_manager.delete_document(result["source_id"])


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
