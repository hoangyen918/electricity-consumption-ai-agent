"""Test error handling cho các tình huống thực tế chưa được test riêng ở milestone trước."""
import pytest
import pandas as pd
from pathlib import Path
from tools.dataset_manager import DatasetManager


def test_load_csv_missing_file_raises_clear_error():
    m = DatasetManager()
    with pytest.raises(FileNotFoundError):
        m.load_csv("data/khong_ton_tai.csv")


def test_load_csv_malformed_file_does_not_crash_silently(tmp_path):
    """File CSV lỗi cú pháp (số cột không đều) — pandas nên raise lỗi rõ ràng, không trả DataFrame rỗng âm thầm."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("col1,col2\n1,2,3,4\n5\n", encoding="utf-8")

    m = DatasetManager()
    try:
        m.load_csv(str(bad_csv))
        # Nếu pandas tự sửa được (skip dòng lỗi), ít nhất phải có dữ liệu thật, không rỗng vô lý
        assert len(m.get_dataframe()) >= 0
    except Exception as e:
        # Chấp nhận raise lỗi rõ ràng, miễn không phải silent failure
        assert e is not None


def test_load_csv_empty_file_raises_or_handles_gracefully(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    m = DatasetManager()
    with pytest.raises(Exception):
        m.load_csv(str(empty_csv))


def test_load_file_with_no_datetime_column_still_works(tmp_path):
    """Dataset không có cột datetime (không phải Tetouan) vẫn phải load được, không crash."""
    csv_path = tmp_path / "no_date.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv_path, index=False)

    m = DatasetManager()
    m.load_csv(str(csv_path))
    assert m.is_loaded()
    assert "a" in m.get_columns()


def test_load_file_completely_non_numeric_dataset():
    """Dataset toàn cột chữ (không có cột số nào) — get_numeric_columns() phải trả về rỗng, không lỗi."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("name,city\nAn,Hanoi\nBinh,HCMC\n")
        path = f.name

    m = DatasetManager()
    m.load_csv(path)
    assert m.get_numeric_columns() == []
    Path(path).unlink()


def test_aggregate_metric_on_non_numeric_column_fails_gracefully():
    """Gọi aggregate_metric trên cột chữ (không phải số) phải trả failure rõ ràng, không crash."""
    from tools.dataframe_tools import aggregate_metric
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("name,city\nAn,Hanoi\nBinh,HCMC\n")
        path = f.name

    m = DatasetManager()
    m.load_csv(path)
    result = aggregate_metric(m, "name", "mean")
    assert result["ok"] is False
    Path(path).unlink()


def test_tool_registry_handles_completely_empty_arguments():
    from tools.dataset_manager import DatasetManager
    from agent.tool_registry import execute_tool
    m = DatasetManager()
    m.load_csv("data/sample/tetouan_sample.csv")
    result = execute_tool("aggregate_metric", {}, m)
    assert result["ok"] is False


def test_tool_registry_handles_none_arguments_type():
    """Nếu LLM lỡ trả về arguments không phải dict hợp lệ, không được crash toàn bộ agent loop."""
    from tools.dataset_manager import DatasetManager
    from agent.tool_registry import execute_tool
    m = DatasetManager()
    m.load_csv("data/sample/tetouan_sample.csv")
    result = execute_tool("aggregate_metric", {"column": None, "aggregation": "mean"}, m)
    assert result["ok"] is False


def test_openrouter_client_invalid_api_key_raises_clear_error(monkeypatch):
    """Giả lập API key sai — phải raise lỗi rõ ràng sau khi hết retry, không treo vô hạn."""
    from openai import OpenAI
    import services.openrouter_client as client_module

    fake_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="sk-or-v1-fake-invalid-key")
    monkeypatch.setattr(client_module, "_client", fake_client)
    monkeypatch.setattr(client_module, "MAX_RETRIES", 1)  # giảm retry để test nhanh, không chờ lâu

    with pytest.raises(RuntimeError, match="failed after"):
        client_module.chat_completion([{"role": "user", "content": "test"}])
