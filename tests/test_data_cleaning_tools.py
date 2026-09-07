import pandas as pd
import pytest

from tools.dataset_manager import DatasetManager
from tools.data_cleaning_tools import (
    clean_missing_values,
    detect_duplicates,
    detect_statistical_anomalies,
)

GT1_PATH_REAL = "data/wind_turbine/data1/his_data_GT1.csv"


def _make_manager_with_missing(tmp_path):
    csv_path = tmp_path / "missing_sample.csv"
    csv_path.write_text(
        "datetime,value_a,value_b\n"
        "2024-01-01 00:00:00,1.0,10\n"
        "2024-01-01 00:05:00,,20\n"
        "2024-01-01 00:10:00,3.0,\n"
        "2024-01-01 00:15:00,4.0,40\n"
        "2024-01-01 00:20:00,,50\n",
        encoding="utf-8",
    )
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m


def test_clean_missing_values_report_counts_nulls_per_column(tmp_path):
    m = _make_manager_with_missing(tmp_path)
    result = clean_missing_values(m, strategy="report")
    assert result["ok"] is True
    assert result["data"]["missing_by_column"]["value_a"]["count"] == 2
    assert result["data"]["missing_by_column"]["value_b"]["count"] == 1
    assert "datetime" not in result["data"]["missing_by_column"]


def test_clean_missing_values_drop_strategy_previews_without_mutating_original(tmp_path):
    m = _make_manager_with_missing(tmp_path)
    rows_before_call = len(m.get_dataframe())

    result = clean_missing_values(m, column="value_a", strategy="drop")

    assert result["ok"] is True
    assert result["data"]["rows_before"] == 5
    assert result["data"]["rows_dropped"] == 2
    assert result["data"]["rows_after"] == 3
    # dataset goc trong manager KHONG bi ghi de
    assert len(m.get_dataframe()) == rows_before_call == 5


def test_clean_missing_values_ffill_reports_correct_fill_count(tmp_path):
    m = _make_manager_with_missing(tmp_path)
    result = clean_missing_values(m, column="value_a", strategy="ffill")
    assert result["ok"] is True
    assert result["data"]["values_filled"] == 2
    assert result["data"]["values_still_null_after"] == 0


def test_clean_missing_values_interpolate_rejects_non_numeric_column(tmp_path):
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text("datetime,label\n2024-01-01,A\n2024-01-02,\n2024-01-03,B\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    result = clean_missing_values(m, column="label", strategy="interpolate")
    assert result["ok"] is False
    assert "không phải kiểu số" in result["error"]


def test_clean_missing_values_invalid_strategy_fails():
    m = DatasetManager()
    result = clean_missing_values(m, strategy="magic")
    assert result["ok"] is False
    assert "not allowed" in result["error"]


def test_detect_duplicates_full_row_and_subset_cols(tmp_path):
    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(
        "ts,a,b\n2024-01-01,1,x\n2024-01-01,1,x\n2024-01-02,2,y\n2024-01-02,9,z\n",
        encoding="utf-8",
    )
    m = DatasetManager()
    m.load_csv(str(csv_path))

    full_row = detect_duplicates(m)
    assert full_row["ok"] is True
    assert full_row["data"]["duplicate_count"] == 1  # chi dong index 1 la ban sao cua dong 0

    subset = detect_duplicates(m, subset_cols=["ts"])
    assert subset["ok"] is True
    assert subset["data"]["duplicate_count"] == 2  # ca 2 cap ts trung deu tinh, tru ban goc


def test_detect_duplicates_invalid_subset_col_fails():
    m = DatasetManager()
    m.load_csv("data/sample/tetouan_sample.csv")
    result = detect_duplicates(m, subset_cols=["khong_ton_tai"])
    assert result["ok"] is False


def test_detect_statistical_anomalies_zscore_detects_outlier(tmp_path):
    csv_path = tmp_path / "outliers.csv"
    rows = ["ts,value"]
    for i in range(20):
        rows.append(f"2024-01-{i+1:02d},10.0")
    rows.append("2024-02-01,500.0")  # outlier ro rang
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = detect_statistical_anomalies(m, column="value", method="zscore", threshold=3.0)
    assert result["ok"] is True
    assert result["data"]["anomaly_count"] == 1
    assert result["data"]["anomaly_values_sample"] == [500.0]


def test_detect_statistical_anomalies_iqr_detects_outlier(tmp_path):
    csv_path = tmp_path / "outliers_iqr.csv"
    rows = ["ts,value"]
    for i in range(20):
        rows.append(f"2024-01-{i+1:02d},10.0")
    rows.append("2024-02-01,500.0")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = detect_statistical_anomalies(m, column="value", method="iqr")
    assert result["ok"] is True
    assert result["data"]["anomaly_count"] == 1


def test_detect_statistical_anomalies_invalid_method_fails():
    m = DatasetManager()
    m.load_csv("data/sample/tetouan_sample.csv")
    result = detect_statistical_anomalies(m, column="temperature", method="not_a_method")
    assert result["ok"] is False
    assert "not allowed" in result["error"]


def test_integration_cleaning_tools_run_on_real_gt1_dataset_without_crash():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")

    report = clean_missing_values(m, dataset_name="GT1", strategy="report")
    assert report["ok"] is True
    assert report["data"]["total_rows"] == 233953
    # cot GT1.App.StartedFlag da biet la thieu nhieu (~34.7%) tu audit
    assert report["data"]["missing_by_column"]["GT1.App.StartedFlag"]["pct"] > 30

    dup = detect_duplicates(m, dataset_name="GT1")
    assert dup["ok"] is True
    assert dup["data"]["duplicate_count"] == 0  # audit da xac nhan 0 dong trung

    anomaly = detect_statistical_anomalies(m, column="GT1.Grid.totW", dataset_name="GT1", method="zscore", threshold=3.0)
    assert anomaly["ok"] is True
    assert 0 <= anomaly["data"]["anomaly_pct"] < 5
