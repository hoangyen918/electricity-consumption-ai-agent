"""Test agent/turbine_tool_wrappers.py — KHÔNG gọi LLM thật, chỉ test logic wrapper
(cache, auto-detect cột, guard đa dataset, giới hạn kích thước output)."""
import pandas as pd
import pytest

from tools.dataset_manager import DatasetManager
from agent.turbine_tool_wrappers import (
    resolve_power_wind_columns,
    get_or_compute_power_curve,
    get_or_compute_classification,
    estimate_power_curve_for_agent,
    classify_turbine_status_for_agent,
    group_anomaly_windows_for_agent,
    predict_power_for_agent,
    TURBINE_DEFAULT_BIN_WIDTH,
    MAX_WINDOWS_FOR_AGENT,
)

GT1_PATH_REAL = "data/wind_turbine/data1/his_data_GT1.csv"
WT1_PATH_REAL = "data/wind_turbine/data2/WindFarm04_WT1.csv"


def _site1_fixture(tmp_path, n=60):
    rows = ["TimeStamp,GT1.Grid.totW,GT1.Common.WindSpd,GT1.SP.ActualSPP"]
    base = pd.Timestamp("2024-01-01")
    for i in range(n):
        ts = base + pd.Timedelta(minutes=10 * i)
        power = 100.0 * (i % 10)
        rows.append(f"{ts.isoformat()},{power},5.5,{power*0.9}")
    csv_path = tmp_path / "site1_fixture.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path), name="site1")
    return m


def test_resolve_power_wind_columns_auto_detects_site1(tmp_path):
    m = _site1_fixture(tmp_path)
    power_col, wind_col, err = resolve_power_wind_columns(m, "site1", None, None)
    assert err is None
    assert power_col == "GT1.Grid.totW"
    assert wind_col == "GT1.Common.WindSpd"


def test_resolve_power_wind_columns_guard_blocks_ambiguous_multi_dataset_call(tmp_path):
    """Phat hien that qua test E2E: LLM doi khi goi tool KHONG dataset_name du co nhieu
    dataset da tai, khien tool ngam chay tren dataset SAI ma khong bao loi. Guard phai
    chan cung, khong chi dua vao prompt engineering."""
    m = _site1_fixture(tmp_path)
    csv2 = tmp_path / "site2.csv"
    csv2.write_text("DATE_TIME,ACTIVE_POWER,WIND_SPEED\n01/01/2024 00:00,100.0,5.0\n", encoding="utf-8")
    m.load_csv(str(csv2), name="site2")

    power_col, wind_col, err = resolve_power_wind_columns(m, None, None, None)
    assert power_col is None and wind_col is None
    assert err is not None
    assert "PHẢI chỉ định rõ dataset_name" in err
    assert "site1" in err and "site2" in err  # liet ke day du dataset hien co

    # nhung neu chi dinh ro dataset_name thi van hoat dong binh thuong
    power_col2, wind_col2, err2 = resolve_power_wind_columns(m, "site1", None, None)
    assert err2 is None
    assert power_col2 == "GT1.Grid.totW"


def test_resolve_power_wind_columns_single_dataset_no_guard_needed(tmp_path):
    """Chi co 1 dataset -> khong can dataset_name ro rang, khong bi guard chan."""
    m = _site1_fixture(tmp_path)
    power_col, wind_col, err = resolve_power_wind_columns(m, None, None, None)
    assert err is None
    assert power_col == "GT1.Grid.totW"


def test_estimate_power_curve_for_agent_caches_and_reuses(tmp_path):
    """Luu y: success() dung dataclasses.asdict() nen deep-copy — khong the kiem tra
    'cung object' (is) qua result['data'], chi kiem tra qua so luong key cache va gia
    tri giong het nhau."""
    m = _site1_fixture(tmp_path)
    r1 = estimate_power_curve_for_agent(m, "site1", None, None, TURBINE_DEFAULT_BIN_WIDTH)
    assert r1["ok"] is True

    cache = getattr(m, "_turbine_power_curve_cache")
    assert len(cache) == 1

    r2 = estimate_power_curve_for_agent(m, "site1", None, None, TURBINE_DEFAULT_BIN_WIDTH)
    assert r2["ok"] is True
    # goi lan 2 voi CUNG tham so KHONG duoc tao them cache entry moi (van tai dung 1 key)
    assert len(cache) == 1
    assert r2["data"] == r1["data"]


def test_classify_turbine_status_for_agent_strips_full_classification(tmp_path):
    m = _site1_fixture(tmp_path)
    result = classify_turbine_status_for_agent(m, "site1", None, None, "GT1.SP.ActualSPP", TURBINE_DEFAULT_BIN_WIDTH)
    assert result["ok"] is True
    assert "classification" not in result["data"]  # KHONG duoc lo dict day du cho LLM
    assert "summary" in result["data"]


def test_predict_power_for_agent_auto_picks_validated_default_features(tmp_path):
    m = _site1_fixture(tmp_path)
    result = predict_power_for_agent(m, "site1", None, None, None, None, TURBINE_DEFAULT_BIN_WIDTH)
    assert result["ok"] is True
    # feature mac dinh cho site1 (da validate that, xem docstring predict_power)
    assert "GT1.Common.WindSpd" in result["data"]["feature_columns"]


def test_group_anomaly_windows_for_agent_caps_output_size(tmp_path):
    """Phat hien that: goi khong thu hep exclude_labels tren du lieu that (GT1) tra ve
    ~700.000 ky tu, vuot context limit model. Wrapper phai tu cap so window tra ve."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")

    # KHONG truyen exclude_labels (mac dinh chi loai Normal) -- day chinh la tinh huong loi that
    result = group_anomaly_windows_for_agent(
        m, "GT1", "datetime", None, None, "GT1.SP.ActualSPP", TURBINE_DEFAULT_BIN_WIDTH, 3, None,
    )
    assert result["ok"] is True
    assert len(result["data"]["windows"]) <= MAX_WINDOWS_FOR_AGENT
    assert result["data"]["windows_truncated"] is True
    assert result["data"]["windows_total_found"] > MAX_WINDOWS_FOR_AGENT

    import json
    serialized_len = len(json.dumps(result, ensure_ascii=False, default=str))
    assert serialized_len < 50_000, f"Ket qua van qua lon sau khi cap: {serialized_len} ky tu"


def test_group_anomaly_windows_for_agent_no_truncation_when_small(tmp_path):
    m = _site1_fixture(tmp_path, n=60)
    result = group_anomaly_windows_for_agent(
        m, "site1", "datetime", None, None, "GT1.SP.ActualSPP", TURBINE_DEFAULT_BIN_WIDTH, 3,
        ["Normal", "Under Production", "Over Production", "Measurement Error", "Stop", "Partial Stop"],
    )
    assert result["ok"] is True
    assert result["data"]["windows_truncated"] is False


# ======================= start_date/end_date time-filter fix =======================
# Phat hien that (bao cao truoc): goi group_anomaly_windows KHONG filter thoi gian tren
# GT1 that -> cap 30-dai-nhat-TOAN-CUC ap dung TRUOC khi biet cau hoi can khoang nao ->
# 5 thang (4,5,6,7,10) bi xoa SACH ket qua du co 60 window that (14+9+10+3+24) trong do.
# CURTAILMENT_EXCLUDE_LABELS chi giu Curtailment/Partial Curtailment (dung nhu khi
# phat hien bug).
CURTAILMENT_EXCLUDE_LABELS = ["Normal", "Measurement Error", "Stop", "Under Production", "Over Production", "Partial Stop"]


def test_group_anomaly_windows_bug_reproduction_default_cap_unchanged_without_date_filter():
    """Tai hien DUNG bug da tim: KHONG truyen start_date/end_date -> hanh vi mac dinh
    KHONG doi, van cap 30-dai-nhat-toan-cuc nhu truoc khi co fix nay."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")

    result = group_anomaly_windows_for_agent(
        m, "GT1", "datetime", None, None, "GT1.SP.ActualSPP", TURBINE_DEFAULT_BIN_WIDTH, 3,
        CURTAILMENT_EXCLUDE_LABELS,
    )
    assert result["ok"] is True
    assert result["data"]["windows_truncated"] is True
    assert len(result["data"]["windows"]) == MAX_WINDOWS_FOR_AGENT
    assert result["data"]["windows_total_found"] == 350  # da xac nhan that qua dieu tra
    assert result["data"]["date_filter_applied"] is None

    # xac nhan dung bug: cac thang co window that (4,5,6,7,10) bi xoa sach khoi ket qua
    months_in_result = {pd.Timestamp(w["start_time"]).month for w in result["data"]["windows"]}
    for missing_month in (4, 5, 6, 7, 10):
        assert missing_month not in months_in_result, (
            f"Thang {missing_month} khong con bi mat nua — hanh vi mac dinh (khong filter) da BI DOI, vi pham rang buoc 'khong doi hanh vi mac dinh'"
        )


def test_group_anomaly_windows_fix_date_filter_recovers_previously_hidden_month():
    """Fix: truyen start_date/end_date cho thang 4/2024 (thang da xac nhan bi xoa sach
    boi cap mac dinh) -> phai thay DAY DU 4 window that cua thang do, khong bi cap mat
    (4 < MAX_WINDOWS_FOR_AGENT_WITH_DATE_FILTER=200 nen khong truncate)."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")

    result = group_anomaly_windows_for_agent(
        m, "GT1", "datetime", None, None, "GT1.SP.ActualSPP", TURBINE_DEFAULT_BIN_WIDTH, 3,
        CURTAILMENT_EXCLUDE_LABELS, start_date="2024-04-01", end_date="2024-04-30",
    )
    assert result["ok"] is True
    assert result["data"]["windows_truncated"] is False
    assert result["data"]["windows_total_found"] == 4  # so that cho DUNG thang 4/2024 (khac 14 = ca thang 4/2024 va 2025 gop lai)
    assert len(result["data"]["windows"]) == 4
    assert result["data"]["date_filter_applied"] == {"start_date": "2024-04-01", "end_date": "2024-04-30"}

    for w in result["data"]["windows"]:
        w_start = pd.Timestamp(w["start_time"])
        assert w_start.year == 2024 and w_start.month == 4


def test_group_anomaly_windows_date_filter_empty_range_returns_empty_not_error():
    """Edge case: khoang thoi gian khong co anomaly nao (hoac ngoai pham vi dataset) ->
    tra ve rong dung nghia, KHONG loi."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")

    result = group_anomaly_windows_for_agent(
        m, "GT1", "datetime", None, None, "GT1.SP.ActualSPP", TURBINE_DEFAULT_BIN_WIDTH, 3,
        CURTAILMENT_EXCLUDE_LABELS, start_date="2030-01-01", end_date="2030-01-31",
    )
    assert result["ok"] is True
    assert result["data"]["windows"] == []
    assert result["data"]["windows_total_found"] == 0
    assert result["data"]["windows_truncated"] is False


def test_group_anomaly_windows_date_filter_invalid_format_fails_gracefully():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    result = group_anomaly_windows_for_agent(
        m, "GT1", "datetime", None, None, "GT1.SP.ActualSPP", TURBINE_DEFAULT_BIN_WIDTH, 3,
        CURTAILMENT_EXCLUDE_LABELS, start_date="khong-phai-ngay-hop-le",
    )
    assert result["ok"] is False
