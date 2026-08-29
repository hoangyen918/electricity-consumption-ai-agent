from pathlib import Path

import pytest

from tools.dataframe_tools import (
    aggregate_metric,
    calculate_correlation,
    compare_periods,
    find_peak,
    get_dataset_summary,
    group_metric,
)
from tools.dataset_manager import DatasetManager

SAMPLE_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample" / "tetouan_sample.csv")
PROCESSED_PATH = str(Path(__file__).resolve().parent.parent / "data" / "processed" / "tetouan_clean.csv")


@pytest.fixture(scope="module")
def manager():
    m = DatasetManager()
    m.load_csv(SAMPLE_PATH)
    return m


@pytest.fixture(scope="module")
def full_manager():
    """
    Dataset đầy đủ cả năm — cần cho regression test end_date, vì SAMPLE_PATH chỉ có
    7 ngày đầu (2017-01-01 đến 2017-01-07), không đủ để test khoảng tháng 7.
    """
    m = DatasetManager()
    m.load_csv(PROCESSED_PATH)
    return m


# --- get_dataset_summary ---


def test_get_dataset_summary_happy_path(manager):
    result = get_dataset_summary(manager)

    assert result["ok"] is True
    assert result["data"]["row_count"] == len(manager.get_dataframe())
    assert "zone_1_consumption" in result["data"]["columns"]
    assert "zone_1_consumption" in result["data"]["numeric_columns"]
    assert result["data"]["date_range"] is not None
    assert "datetime" in result["data"]["dtypes"]


# --- aggregate_metric ---


def test_aggregate_metric_happy_path(manager):
    result = aggregate_metric(manager, column="temperature", aggregation="mean")

    assert result["ok"] is True
    assert isinstance(result["data"]["value"], float)
    assert result["data"]["column"] == "temperature"
    assert result["data"]["aggregation"] == "mean"
    assert result["data"]["rows_used"] == len(manager.get_dataframe())


def test_aggregate_metric_invalid_column(manager):
    result = aggregate_metric(manager, column="not_a_real_column", aggregation="mean")

    assert result["ok"] is False
    assert "not_a_real_column" in result["error"]


def test_aggregate_metric_invalid_aggregation(manager):
    result = aggregate_metric(manager, column="temperature", aggregation="bogus_agg")

    assert result["ok"] is False
    assert "bogus_agg" in result["error"]


def test_aggregate_metric_empty_date_range(manager):
    result = aggregate_metric(
        manager,
        column="temperature",
        aggregation="mean",
        start_date="1999-01-01",
        end_date="1999-01-02",
    )

    assert result["ok"] is False
    assert result["data"] is None


# --- group_metric ---


def test_group_metric_happy_path(manager):
    result = group_metric(manager, column="zone_1_consumption", aggregation="mean", group_by="hour")

    assert result["ok"] is True
    assert result["data"]["group_by"] == "hour"
    assert isinstance(result["data"]["values"], dict)
    assert len(result["data"]["values"]) > 0
    keys_as_int = [int(k) for k in result["data"]["values"].keys()]
    assert keys_as_int == sorted(keys_as_int)


def test_group_metric_invalid_column(manager):
    result = group_metric(manager, column="not_a_real_column", aggregation="mean", group_by="hour")

    assert result["ok"] is False
    assert "not_a_real_column" in result["error"]


def test_group_metric_invalid_group_by(manager):
    result = group_metric(manager, column="zone_1_consumption", aggregation="mean", group_by="second")

    assert result["ok"] is False
    assert "second" in result["error"]


# --- find_peak ---


def test_find_peak_happy_path(manager):
    result = find_peak(manager, column="temperature", mode="max")

    df = manager.get_dataframe()
    expected_max = df["temperature"].max()

    assert result["ok"] is True
    assert result["data"]["value"] == pytest.approx(expected_max)
    assert result["data"]["mode"] == "max"
    assert result["data"]["datetime"] is not None


def test_find_peak_invalid_column(manager):
    result = find_peak(manager, column="not_a_real_column")

    assert result["ok"] is False
    assert "not_a_real_column" in result["error"]


def test_find_peak_invalid_mode(manager):
    result = find_peak(manager, column="temperature", mode="middle")

    assert result["ok"] is False
    assert "middle" in result["error"]


# --- compare_periods ---


def test_compare_periods_happy_path(manager):
    result = compare_periods(
        manager,
        column="zone_1_consumption",
        period_a_start="2017-01-01",
        period_a_end="2017-01-02",
        period_b_start="2017-01-03",
        period_b_end="2017-01-04",
    )

    assert result["ok"] is True
    assert result["data"]["period_a"]["rows_used"] > 0
    assert result["data"]["period_b"]["rows_used"] > 0
    assert result["data"]["absolute_difference"] is not None


def test_compare_periods_invalid_column(manager):
    result = compare_periods(
        manager,
        column="not_a_real_column",
        period_a_start="2017-01-01",
        period_a_end="2017-01-02",
        period_b_start="2017-01-03",
        period_b_end="2017-01-04",
    )

    assert result["ok"] is False
    assert "not_a_real_column" in result["error"]


def test_compare_periods_invalid_aggregation(manager):
    result = compare_periods(
        manager,
        column="zone_1_consumption",
        period_a_start="2017-01-01",
        period_a_end="2017-01-02",
        period_b_start="2017-01-03",
        period_b_end="2017-01-04",
        aggregation="bogus_agg",
    )

    assert result["ok"] is False
    assert "bogus_agg" in result["error"]


def test_compare_periods_empty_date_range(manager):
    result = compare_periods(
        manager,
        column="zone_1_consumption",
        period_a_start="1999-01-01",
        period_a_end="1999-01-02",
        period_b_start="2017-01-03",
        period_b_end="2017-01-04",
    )

    assert result["ok"] is False
    assert result["data"] is None


# --- calculate_correlation ---


def test_calculate_correlation_happy_path(manager):
    result = calculate_correlation(manager, column_a="temperature", column_b="humidity")

    assert result["ok"] is True
    assert -1.0 <= result["data"]["correlation_coefficient"] <= 1.0
    assert result["data"]["interpretation"] in (
        "strong positive",
        "moderate positive",
        "weak/no correlation",
        "moderate negative",
        "strong negative",
    )


def test_calculate_correlation_invalid_column(manager):
    result = calculate_correlation(manager, column_a="not_a_real_column", column_b="humidity")

    assert result["ok"] is False
    assert "not_a_real_column" in result["error"]


# --- regression: end_date phải bao gồm TOÀN BỘ ngày cuối, không chỉ đúng lúc 00:00 ---


def test_aggregate_metric_end_date_includes_full_last_day(manager):
    """
    Regression test cho bug đã phát hiện: end_date phải bao gồm TOÀN BỘ ngày cuối
    (144 dòng/ngày với dữ liệu 10 phút/lần), không chỉ đúng 1 dòng lúc 00:00.
    """
    result = aggregate_metric(manager, "zone_1_consumption", "count",
                               start_date="2017-01-01", end_date="2017-01-01")
    assert result["ok"] is True
    assert result["data"]["rows_used"] == 144, (
        f"Ngày 2017-01-01 phải có đủ 144 dòng (10 phút/lần), "
        f"chỉ thấy {result['data']['rows_used']} — bug end_date filtering đã tái diễn"
    )


def test_aggregate_metric_multi_day_range_includes_all_days(full_manager):
    """Khoảng nhiều ngày phải bao gồm đủ ngày cuối, không thiếu 143 dòng như bug cũ."""
    result = aggregate_metric(full_manager, "zone_1_consumption", "count",
                               start_date="2017-07-01", end_date="2017-07-31")
    assert result["ok"] is True
    assert result["data"]["rows_used"] == 31 * 144, (
        f"Tháng 7/2017 (31 ngày x 144 dòng/ngày = 4464) phải đủ, "
        f"chỉ thấy {result['data']['rows_used']}"
    )
