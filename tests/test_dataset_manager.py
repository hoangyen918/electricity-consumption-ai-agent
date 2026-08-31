from pathlib import Path

import pandas as pd
import pytest

from tools.dataset_manager import DatasetManager
from tools.dataframe_tools import group_metric

SAMPLE_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample" / "tetouan_sample.csv")


def test_load_csv_success():
    manager = DatasetManager()
    manager.load_csv(SAMPLE_PATH)

    assert manager.is_loaded() is True
    df = manager.get_dataframe()
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])


def test_get_dataframe_raises_when_not_loaded():
    manager = DatasetManager()

    with pytest.raises(RuntimeError):
        manager.get_dataframe()


def test_get_columns_returns_correct_list():
    manager = DatasetManager()
    manager.load_csv(SAMPLE_PATH)

    columns = manager.get_columns()

    assert "datetime" in columns
    assert "zone_1_consumption" in columns
    assert "temperature" in columns
    assert columns == list(manager.get_dataframe().columns)


def test_get_numeric_columns_excludes_datetime():
    manager = DatasetManager()
    manager.load_csv(SAMPLE_PATH)

    numeric_columns = manager.get_numeric_columns()

    assert "datetime" not in numeric_columns
    assert "zone_1_consumption" in numeric_columns
    assert "temperature" in numeric_columns


def test_load_file_dispatches_to_csv():
    m = DatasetManager()
    m.load_file("data/sample/tetouan_sample.csv")
    assert m.is_loaded()
    assert "zone_1_consumption" in m.get_columns()


def test_load_file_rejects_unsupported_extension():
    m = DatasetManager()
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        m.load_file("some_file.json")


def test_load_excel_reads_correctly(tmp_path):
    """Tạo 1 file Excel tạm từ chính sample CSV để test load_excel không cần file Excel thật có sẵn."""
    import pandas as pd
    df = pd.read_csv("data/sample/tetouan_sample.csv")
    excel_path = tmp_path / "test_sample.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")

    m = DatasetManager()
    m.load_excel(str(excel_path))
    assert m.is_loaded()
    assert len(m.get_dataframe()) == len(df)
    assert "zone_1_consumption" in m.get_columns()


def test_upload_of_raw_tetouan_style_csv_supports_group_by_month(tmp_path):
    """
    Regression test cho bug đã phát hiện: dataset upload qua UI (chưa qua
    scripts/prepare_data.py, giữ nguyên tên cột gốc kiểu 'DateTime', 'Zone 1 Power
    Consumption') phải vẫn dùng được group_metric(group_by='month') nhờ
    derive_time_columns() tự phát hiện cột datetime và derive cột 'month' -
    KHÔNG rename theo RENAME_MAP dành riêng cho Tetouan, cột giá trị vẫn dùng
    đúng tên gốc như trong dataset.
    """
    raw_csv = tmp_path / "tetouan_raw_like.csv"
    rows = []
    for month in range(1, 13):
        for day in (1, 15):
            rows.append(f"2017-{month:02d}-{day:02d} 10:00:00,20.0,55.0,{month}00.0")
    content = "DateTime,Temperature,Humidity,Zone 1 Power Consumption\n" + "\n".join(rows) + "\n"
    raw_csv.write_text(content, encoding="utf-8")

    manager = DatasetManager()
    manager.load_csv(str(raw_csv))

    assert manager.get_datetime_source_column() == "DateTime"
    assert "Zone 1 Power Consumption" in manager.get_columns()

    result = group_metric(
        manager, column="Zone 1 Power Consumption", aggregation="mean", group_by="month"
    )

    assert result["ok"] is True, result.get("error")
    assert len(result["data"]["values"]) == 12
    # month=7 -> giá trị Zone 1 Power Consumption luôn là 700.0 theo dữ liệu tạo ở trên
    assert result["data"]["values"]["7"] == pytest.approx(700.0)
