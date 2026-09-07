import pandas as pd
import pytest

from tools.dataset_manager import DatasetManager
from tools.data_cleaning_tools import (
    clean_missing_values,
    detect_duplicates,
    detect_statistical_anomalies,
    correlation_analysis,
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


def _correlation_fixture(tmp_path):
    csv_path = tmp_path / "corr_sample.csv"
    rows = ["ts,a,b_strong_pos,c_strong_neg,d_weak,label"]
    for i in range(30):
        a = float(i)
        b = a * 2 + 1  # tuong quan duong hoan hao voi a
        c = -a * 3 + 5  # tuong quan am hoan hao voi a
        d = (i % 3) * 0.1  # gan nhu khong lien quan a
        rows.append(f"2024-01-{(i%28)+1:02d},{a},{b},{c},{d},X")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m


def test_correlation_analysis_matches_manual_pandas_corr(tmp_path):
    m = _correlation_fixture(tmp_path)
    df = m.get_dataframe()
    expected_b = df["a"].corr(df["b_strong_pos"])
    expected_c = df["a"].corr(df["c_strong_neg"])

    result = correlation_analysis(m, target_column="a", top_n=10)
    assert result["ok"] is True
    by_col = {c["column"]: c["correlation"] for c in result["data"]["correlations"]}
    assert by_col["b_strong_pos"] == pytest.approx(round(expected_b, 4), abs=1e-4)
    assert by_col["c_strong_neg"] == pytest.approx(round(expected_c, 4), abs=1e-4)


def test_correlation_analysis_sorts_by_abs_value_but_keeps_sign(tmp_path):
    m = _correlation_fixture(tmp_path)
    result = correlation_analysis(m, target_column="a", top_n=10)
    assert result["ok"] is True
    correlations = result["data"]["correlations"]

    # b_strong_pos (~+1.0) va c_strong_neg (~-1.0) phai dung dau va dung o dau (|corr| lon nhat)
    top_cols = [c["column"] for c in correlations[:2]]
    assert set(top_cols) == {"b_strong_pos", "c_strong_neg"}
    b_entry = next(c for c in correlations if c["column"] == "b_strong_pos")
    c_entry = next(c for c in correlations if c["column"] == "c_strong_neg")
    assert b_entry["correlation"] > 0
    assert c_entry["correlation"] < 0

    # d_weak phai co |corr| nho hon han b_strong_pos/c_strong_neg (khong assert vi tri
    # tuyet doi cuoi cung, vi dataset con co cac cot thoi gian tu derive_time_columns
    # nhu 'weekday' co the |corr| con nho hon nua voi 'a' — day la hanh vi dung cua
    # tool, khong phai bug)
    d_entry = next(c for c in correlations if c["column"] == "d_weak")
    assert abs(d_entry["correlation"]) < abs(b_entry["correlation"])
    assert abs(d_entry["correlation"]) < abs(c_entry["correlation"])


def test_correlation_analysis_top_n_limits_results(tmp_path):
    m = _correlation_fixture(tmp_path)
    df = m.get_dataframe()
    expected_total = len(df.select_dtypes(include="number").columns) - 1  # tru chinh cot 'a'

    result = correlation_analysis(m, target_column="a", top_n=1)
    assert result["ok"] is True
    assert len(result["data"]["correlations"]) == 1
    assert result["data"]["total_columns_compared"] == expected_total


def test_correlation_analysis_rejects_non_numeric_target(tmp_path):
    m = _correlation_fixture(tmp_path)
    result = correlation_analysis(m, target_column="label")
    assert result["ok"] is False
    assert "không phải kiểu số" in result["error"]


def test_integration_correlation_analysis_on_real_gt1_values_in_valid_range():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    result = correlation_analysis(m, target_column="GT1.Grid.totW", dataset_name="GT1", top_n=5)
    assert result["ok"] is True
    for c in result["data"]["correlations"]:
        assert -1.0 <= c["correlation"] <= 1.0
    # GT1.Common.TheoActiveW ly thuyet phai tuong quan rat manh voi Grid.totW thuc te
    top_col_names = [c["column"] for c in result["data"]["correlations"]]
    assert "GT1.Common.TheoActiveW" in top_col_names
