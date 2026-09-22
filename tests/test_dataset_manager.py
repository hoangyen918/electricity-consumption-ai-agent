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


def test_multi_dataset_load_and_retrieve_by_name(tmp_path):
    """Upload 2 dataset khác schema -> cả 2 cùng tồn tại, get_dataframe(name) trả đúng
    df tương ứng, không ghi đè nhau."""
    other_csv = tmp_path / "other_schema.csv"
    other_csv.write_text("colA,colB\n1,2\n3,4\n", encoding="utf-8")

    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="tetouan.csv")
    m.load_csv(str(other_csv), name="other_schema.csv")

    assert set(m.list_datasets()) == {"tetouan.csv", "other_schema.csv"}
    assert m.get_active_name() == "other_schema.csv"  # file load gần nhất

    df1 = m.get_dataframe("tetouan.csv")
    df2 = m.get_dataframe("other_schema.csv")
    assert "zone_1_consumption" in df1.columns
    assert list(df2.columns) == ["colA", "colB"]
    assert len(df2) == 2


def test_get_dataframe_nonexistent_name_returns_none():
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="only_one.csv")
    assert m.get_dataframe("khong_ton_tai.csv") is None


def test_remove_dataset_switches_active_to_remaining():
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="a.csv")
    m.load_csv(SAMPLE_PATH, name="b.csv")
    assert m.get_active_name() == "b.csv"

    assert m.remove_dataset("b.csv") is True
    assert m.get_active_name() == "a.csv"
    assert m.list_datasets() == ["a.csv"]

    assert m.remove_dataset("a.csv") is True
    assert m.get_active_name() is None
    assert m.list_datasets() == []


def test_remove_nonexistent_dataset_returns_false():
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="a.csv")
    assert m.remove_dataset("khong_ton_tai") is False


def test_switch_dataset_changes_active_without_reloading():
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="a.csv")
    m.load_csv(SAMPLE_PATH, name="b.csv")
    assert m.get_active_name() == "b.csv"

    assert m.switch_dataset("a.csv") is True
    assert m.get_active_name() == "a.csv"
    # switch khong load lai file - danh sach dataset khong doi
    assert m.list_datasets() == ["a.csv", "b.csv"]


def test_switch_dataset_nonexistent_returns_false_keeps_active():
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="a.csv")
    assert m.switch_dataset("khong_ton_tai") is False


def test_load_csv_set_active_false_keeps_existing_active():
    """set_active=False khi manager ĐÃ có active — active KHÔNG đổi, dataset mới vẫn
    được thêm vào (get_dataframe(tên đó) vẫn truy cập được bình thường)."""
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="a.csv")
    assert m.get_active_name() == "a.csv"

    m.load_csv(SAMPLE_PATH, name="b.csv", set_active=False)

    assert m.get_active_name() == "a.csv"
    assert m.list_datasets() == ["a.csv", "b.csv"]
    assert m.get_dataframe("b.csv") is not None


def test_add_dataframe_set_active_false_keeps_existing_active():
    """Cùng hành vi test_load_csv_set_active_false_keeps_existing_active nhưng qua
    add_dataframe() — đây là method THẬT SỰ được dùng khi bridge turbine dataset."""
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="a.csv")
    assert m.get_active_name() == "a.csv"

    other_df = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    m.add_dataframe(other_df, name="b.csv", set_active=False)

    assert m.get_active_name() == "a.csv"
    assert m.list_datasets() == ["a.csv", "b.csv"]


def test_load_csv_set_active_false_on_empty_manager_keeps_active_none():
    """Edge case CỐ Ý (xem docstring load_csv()): manager rỗng (active=None) + nạp với
    set_active=False -> active vẫn None sau đó, dataset tồn tại nhưng
    get_dataframe(None) vẫn raise RuntimeError cho tới khi có 1 lần nạp/switch active
    thật — KHÔNG tự động "nhận" active dù đó là dataset đầu tiên."""
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH, name="only.csv", set_active=False)

    assert m.get_active_name() is None
    assert m.list_datasets() == ["only.csv"]
    assert m.get_dataframe("only.csv") is not None
    with pytest.raises(RuntimeError):
        m.get_dataframe()


def test_load_csv_auto_detects_semicolon_delimiter_with_bom(tmp_path):
    """
    Regression test cho bug phát hiện qua dataset wind turbine WindFarm04_WT*.csv:
    file dùng dấu chấm phẩy làm delimiter + có BOM utf-8 ở đầu file (xuất từ hệ
    thống châu Âu) - trước khi sửa, pd.read_csv() mặc định dấu phẩy khiến toàn bộ
    dòng bị gộp thành 1 cột duy nhất, và BOM dính vào tên cột đầu tiên.
    """
    semicolon_csv = tmp_path / "semicolon_bom.csv"
    content = "DATE_TIME;ACTIVE_POWER;WIND_SPEED\n01/01/2012 00:00;1174.61;9.21\n13/01/2012 00:10;1072.58;8.86\n"
    semicolon_csv.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

    m = DatasetManager()
    m.load_csv(str(semicolon_csv))

    df = m.get_dataframe()
    assert list(df.columns[:3]) == ["DATE_TIME", "ACTIVE_POWER", "WIND_SPEED"]
    assert len(df) == 2
    assert df.iloc[0]["ACTIVE_POWER"] == pytest.approx(1174.61)
    # dòng thứ 2 co ngay > 12 -> phai duoc hieu la DD/MM/YYYY (dayfirst=True)
    assert df.iloc[1]["day"] == 13
    assert df.iloc[1]["month"] == 1


def test_load_csv_still_uses_comma_delimiter_for_existing_datasets(tmp_path):
    """Đảm bảo _detect_delimiter() không phá vỡ dataset dấu phẩy hiện có (vd Tetouan) -
    delimiter tự phát hiện phải vẫn ra dấu phẩy như hành vi cũ."""
    comma_csv = tmp_path / "comma_plain.csv"
    comma_csv.write_text("colA,colB,colC\n1,2,3\n4,5,6\n", encoding="utf-8")

    m = DatasetManager()
    m.load_csv(str(comma_csv))

    df = m.get_dataframe()
    assert list(df.columns) == ["colA", "colB", "colC"]
    assert len(df) == 2
