import pandas as pd

from tools.dataset_manager import derive_time_columns


def test_detects_datetime_named_column_tetouan_style():
    """Cột tên 'DateTime' (giống Tetouan) có timestamp hợp lệ -> detect đúng, derive đủ 6 cột."""
    df = pd.DataFrame({
        "DateTime": ["2017-01-05 13:20:00", "2017-07-15 08:00:00", "2017-12-31 23:50:00"],
        "Zone 1 Power Consumption": [100.0, 200.0, 300.0],
    })

    result, source_column = derive_time_columns(df)

    assert source_column == "DateTime"
    for col in ["year", "month", "day", "hour", "weekday", "date"]:
        assert col in result.columns

    assert result["year"].tolist() == [2017, 2017, 2017]
    assert result["month"].tolist() == [1, 7, 12]
    assert result["day"].tolist() == [5, 15, 31]
    assert result["hour"].tolist() == [13, 8, 23]
    # original column giữ nguyên, không bị đổi tên/xóa
    assert "DateTime" in result.columns
    assert result["DateTime"].tolist() == df["DateTime"].tolist()


def test_detects_arbitrary_column_name_not_hardcoded():
    """Cột tên hoàn toàn khác ('fecha') chứa ISO timestamp -> vẫn phải detect đúng
    (bằng chứng cho yêu cầu chạy được trên CSV lạ, không hard-code tên cột)."""
    df = pd.DataFrame({
        "fecha": ["2020-03-01T00:00:00", "2020-03-02T00:00:00", "2020-03-03T00:00:00"],
        "consumo": [10.5, 11.2, 9.8],
    })

    result, source_column = derive_time_columns(df)

    assert source_column == "fecha"
    assert "datetime" in result.columns
    assert result["month"].tolist() == [3, 3, 3]
    assert result["day"].tolist() == [1, 2, 3]


def test_numeric_column_excluded_even_if_values_look_like_dates():
    """Cột int64 (ví dụ 'year' toàn giá trị 2017) PHẢI bị loại vì là cột numeric,
    không được nhận nhầm làm datetime."""
    df = pd.DataFrame({
        "year": [2017, 2017, 2017, 2017],
        "reading": [1.1, 2.2, 3.3, 4.4],
    })

    result, source_column = derive_time_columns(df)

    assert source_column is None
    assert "datetime" not in result.columns
    assert "month" not in result.columns
    assert result["year"].tolist() == [2017, 2017, 2017, 2017]


def test_below_threshold_parse_rate_rejected():
    """Cột chuỗi chỉ 80% dòng parse được (20% rác) -> dưới ngưỡng 95%, PHẢI bị từ
    chối, không derive cột nào."""
    values = ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04",
              "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08",
              "khong_phai_ngay", "N/A"]
    assert len(values) == 10
    df = pd.DataFrame({"ngay_gio": values, "value": list(range(10))})

    result, source_column = derive_time_columns(df)

    assert source_column is None
    assert "datetime" not in result.columns
    for col in ["year", "month", "day", "hour", "weekday", "date"]:
        assert col not in result.columns


def test_existing_datetime_and_month_columns_not_overwritten():
    """Dataset đã có sẵn cột 'datetime' hợp lệ + đã có sẵn 'month' -> không bị
    derive đè/tính lại, giữ nguyên giá trị cũ."""
    df = pd.DataFrame({
        "datetime": ["2017-01-05 13:20:00", "2017-01-06 14:20:00"],
        "month": [99, 99],  # giá trị cố tình sai để chứng minh không bị tính lại
        "zone_1_consumption": [100.0, 110.0],
    })

    result, source_column = derive_time_columns(df)

    assert source_column == "datetime"
    assert result["month"].tolist() == [99, 99]
    # datetime dtype vẫn được chuẩn hóa đúng (fix bug case-sensitivity/parse cũ)
    assert pd.api.types.is_datetime64_any_dtype(result["datetime"])
    # các cột derived còn thiếu vẫn được thêm bình thường
    for col in ["year", "day", "hour", "weekday", "date"]:
        assert col in result.columns
    assert result["year"].tolist() == [2017, 2017]


def test_dmy_slash_format_detected_via_dayfirst():
    """
    Regression test cho bug thật phát hiện qua test Docker (logs/csv_docker_test_20260901.log,
    schema giống solar_station_unfamiliar.csv): cột ngày dạng 'DD/MM/YYYY' (phổ biến ở VN) bị
    derive_time_columns() TỪ CHỐI hoàn toàn trước khi fix, vì pd.to_datetime() mặc định hiểu
    theo MM/DD (dayfirst=False) -- các ngày >12 (15, 20, 25) khiến "tháng" invalid -> coerce
    thành NaT, tỷ lệ parse thành công chỉ 5/8 = 62.5%, dưới ngưỡng 95%.

    Sau fix (_parse_datetime_best_effort thử cả dayfirst=False lẫn True, chọn tỷ lệ cao hơn):
    dayfirst=True cho tỷ lệ 8/8 = 100%, cột được detect đúng.
    """
    df = pd.DataFrame({
        "ngay_ghi_nhan": [
            "01/03/2024", "02/03/2024", "03/03/2024", "15/03/2024",
            "20/03/2024", "01/04/2024", "10/04/2024", "25/04/2024",
        ],
        "san_luong_kwh": [152.3, 148.7, 160.1, 133.4, 155.9, 140.2, 165.8, 138.6],
    })

    result, source_column = derive_time_columns(df)

    assert source_column == "ngay_ghi_nhan"
    assert "datetime" in result.columns
    assert result["month"].tolist() == [3, 3, 3, 3, 3, 4, 4, 4]
    assert result["day"].tolist() == [1, 2, 3, 15, 20, 1, 10, 25]
    assert result["year"].tolist() == [2024] * 8


def test_no_datetime_like_column_at_all_no_exception():
    """Dataset hoàn toàn không có cột nào giống ngày giờ (toàn số) -> không derive
    gì, không raise exception."""
    df = pd.DataFrame({
        "temperature": [20.1, 21.5, 19.8],
        "humidity": [50, 55, 60],
    })

    result, source_column = derive_time_columns(df)

    assert source_column is None
    assert list(result.columns) == ["temperature", "humidity"]
