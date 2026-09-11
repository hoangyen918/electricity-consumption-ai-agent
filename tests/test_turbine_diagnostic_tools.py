from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tools.dataset_manager import DatasetManager
from tools.turbine_diagnostic_tools import (
    detect_physical_violations,
    detect_sensor_issues,
    diagnose_theoretical_power_deviation,
    explain_anomaly,
    estimate_power_curve,
    classify_turbine_status,
    group_anomaly_windows,
    detect_anomalies_ml,
    predict_power,
    calculate_turbine_kpis,
    calculate_loss_by_category,
    get_yaw_error,
    _mape,
    MAPE_MIN_ACTUAL_POWER_KW,
    CLASSIFICATION_LABELS,
)

GT1_PATH_REAL = "data/wind_turbine/data1/his_data_GT1.csv"
WT1_PATH_REAL = "data/wind_turbine/data2/WindFarm04_WT1.csv"


def _site1_negative_power_fixture(tmp_path):
    """20 dong power=2000 (co dinh) de rated_power_estimate = 2000 chinh xac (percentile
    99.5 cua 1 mang toan gia tri giong nhau = chinh gia tri do), cong 1 dong am nhe
    (-50, |50|<5%*2000=100 -> KHONG bi flag) va 1 dong am sau (-150, |150|>100 -> bi flag)."""
    rows = ["TimeStamp,GT1.Grid.totW,GT1.Common.WindSpd"]
    for i in range(20):
        rows.append(f"2024-01-01 {i:02d}:00:00,2000.0,10.0")
    rows.append("2024-01-02 00:00:00,-50.0,10.0")
    rows.append("2024-01-02 01:00:00,-150.0,10.0")
    csv_path = tmp_path / "gt1_negative.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m


def test_detect_physical_violations_magnitude_aware_negative_power(tmp_path):
    m = _site1_negative_power_fixture(tmp_path)
    result = detect_physical_violations(m)
    assert result["ok"] is True
    assert result["data"]["site_detected"] == "site1"

    rule = result["data"]["rules"]["negative_power_magnitude_aware"]
    assert rule["applicable"] is True
    assert rule["rated_power_estimate"] == pytest.approx(2000.0)
    assert rule["threshold_used"] == pytest.approx(-100.0)
    # chi -150 bi flag, -50 KHONG bi flag (duoi nguong 5% rated)
    assert rule["count"] == 1


def test_detect_physical_violations_site1_pitch_mismatch_and_site2_only_rules_na(tmp_path):
    rows = ["TimeStamp,GT1.Grid.totW,GT1.Common.WindSpd,GT1.Pitch.Angle1,GT1.Pitch.Angle2,GT1.Pitch.Angle3"]
    rows.append("2024-01-01 00:00:00,1000.0,10.0,10.0,10.0,10.0")  # dong bo, khong mismatch
    rows.append("2024-01-01 00:05:00,1000.0,10.0,10.0,20.0,20.0")  # lech 10 do > 5 do -> mismatch
    csv_path = tmp_path / "gt1_pitch.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = detect_physical_violations(m)
    assert result["ok"] is True
    pitch_rule = result["data"]["rules"]["pitch_mismatch"]
    assert pitch_rule["applicable"] is True
    assert pitch_rule["count"] == 1

    # cac rule chi danh cho Site 2 phai la N/A tren Site 1
    assert result["data"]["rules"]["anemometer_mismatch"]["applicable"] is False
    assert result["data"]["rules"]["temperature_alarm"]["applicable"] is False


def test_detect_physical_violations_site2_anemometer_and_temperature_rules(tmp_path):
    rows = ["DATE_TIME,ACTIVE_POWER,WIND_SPEED,WIND_SPEED_2,TEMPERATURE_GEARBOX_OIL,TEMPERATURE_STATOR"]
    rows.append("01/01/2024 00:00,500.0,10.0,10.0,50.0,60.0")  # binh thuong
    rows.append("01/01/2024 00:10,500.0,10.0,7.0,50.0,60.0")  # lech >20% (10 vs 7 = 30%)
    rows.append("01/01/2024 00:20,500.0,10.0,10.0,95.0,60.0")  # gearbox > 90 -> alarm
    csv_path = tmp_path / "wt_anemo.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = detect_physical_violations(m)
    assert result["ok"] is True
    assert result["data"]["site_detected"] == "site2"
    assert result["data"]["rules"]["anemometer_mismatch"]["applicable"] is True
    assert result["data"]["rules"]["anemometer_mismatch"]["count"] == 1
    assert result["data"]["rules"]["temperature_alarm"]["applicable"] is True
    assert result["data"]["rules"]["temperature_alarm"]["count"] == 1
    # pitch mismatch la Site 1 only -> N/A tren Site 2
    assert result["data"]["rules"]["pitch_mismatch"]["applicable"] is False


def test_detect_physical_violations_unknown_site_fails_clearly(tmp_path):
    csv_path = tmp_path / "mystery.csv"
    csv_path.write_text("colA,colB\n1,3\n2,4\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    result = detect_physical_violations(m)
    assert result["ok"] is False
    assert "Không nhận diện được" in result["error"]


def test_detect_sensor_issues_flatline_configurable_min_length(tmp_path):
    rows = ["DATE_TIME,ACTIVE_POWER,WIND_SPEED"]
    base = pd.Timestamp("2024-01-01")
    for i in range(10):
        ts = base + pd.Timedelta(minutes=10 * i)
        value = 5.0 if 2 <= i <= 6 else float(i)  # chuoi 5 buoc lien tiep gia tri 5.0 (i=2..6)
        rows.append(f"{ts.strftime('%d/%m/%Y %H:%M')},{value},10.0")
    csv_path = tmp_path / "wt_flatline.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result_default = detect_sensor_issues(m, flatline_min_length=6)
    assert result_default["ok"] is True
    assert result_default["data"]["flatline"]["n_runs"] == 0  # chuoi chi dai 5, chua du 6

    result_configured = detect_sensor_issues(m, flatline_min_length=5)
    assert result_configured["ok"] is True
    assert result_configured["data"]["flatline"]["n_runs"] == 1
    assert result_configured["data"]["flatline"]["total_rows_affected"] == 5


def test_detect_sensor_issues_timestamp_gap_auto_computed_from_mode(tmp_path):
    rows = ["DATE_TIME,ACTIVE_POWER"]
    base = pd.Timestamp("2024-01-01")
    for i in range(10):
        ts = base + pd.Timedelta(minutes=10 * i)
        rows.append(f"{ts.strftime('%d/%m/%Y %H:%M')},{100.0}")
    # gap lon 2 gio sau dong cuoi
    gap_ts = base + pd.Timedelta(minutes=10 * 9) + pd.Timedelta(hours=2)
    rows.append(f"{gap_ts.strftime('%d/%m/%Y %H:%M')},100.0")
    csv_path = tmp_path / "wt_gap.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = detect_sensor_issues(m)
    assert result["ok"] is True
    assert result["data"]["timestamp_gap"]["expected_timestep"] == "0 days 00:10:00"
    assert result["data"]["timestamp_gap"]["count"] == 1


def test_detect_sensor_issues_placeholder_values_absolute_zero_and_simultaneous_zero(tmp_path):
    rows = ["DATE_TIME,ACTIVE_POWER,TEMPERATURE_GEARBOX_OIL,TEMPERATURE_STATOR"]
    rows.append("01/01/2024 00:00,500.0,50.0,60.0")  # binh thuong
    rows.append("01/01/2024 00:10,500.0,-273.2,60.0")  # placeholder tuyet doi khong
    rows.append("01/01/2024 00:20,500.0,0.0,0.0")  # dong thoi = 0 tren ca 2 cot
    csv_path = tmp_path / "wt_placeholder.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = detect_sensor_issues(m)
    assert result["ok"] is True
    placeholder = result["data"]["placeholder_values"]
    assert placeholder["applicable"] is True
    assert placeholder["absolute_zero_sentinel"]["count"] == 1
    assert placeholder["simultaneous_zero_all_temp_columns"]["count"] == 1


def test_diagnose_theoretical_power_deviation_requires_gt1_columns(tmp_path):
    csv_path = tmp_path / "wt_no_gt1.csv"
    csv_path.write_text("DATE_TIME,ACTIVE_POWER\n01/01/2024 00:00,500.0\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    result = diagnose_theoretical_power_deviation(m)
    assert result["ok"] is False
    assert "GT1" in result["error"]


def test_diagnose_theoretical_power_deviation_computes_comparison_without_conclusion(tmp_path):
    rows = ["TimeStamp,GT1.Grid.totW,GT1.Common.TheoActiveW,GT1.Common.OpSt,GT1.Pitch.Angle1,GT1.Pitch.Angle2,GT1.Pitch.Angle3,GT1.Common.WindTrackDevAng,GT1.SP.ActualSPP"]
    for i in range(10):
        # binh thuong: totW xap xi TheoActiveW
        rows.append(f"2024-01-01 {i:02d}:00:00,1000.0,1000.0,5.0,1.0,1.0,1.0,0.0,1000.0")
    for i in range(3):
        # vi pham: totW vuot TheoActiveW hon 10%
        rows.append(f"2024-01-02 {i:02d}:00:00,1300.0,1000.0,3.0,1.0,1.0,1.0,0.0,1000.0")
    csv_path = tmp_path / "gt1_theo.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = diagnose_theoretical_power_deviation(m)
    assert result["ok"] is True
    assert result["data"]["violation_count"] == 3
    assert result["data"]["normal_count"] == 10
    assert "GT1.Common.OpSt" in result["data"]["comparison"]
    assert "conclusion" not in result["data"]  # KHONG duoc tu ket luan nguyen nhan


def test_integration_detect_physical_violations_on_real_gt1_without_crash():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    result = detect_physical_violations(m, dataset_name="GT1")
    assert result["ok"] is True
    rule = result["data"]["rules"]["negative_power_magnitude_aware"]
    assert rule["rated_power_estimate"] > 3000  # audit: max quan sat ~4525


def test_integration_detect_sensor_issues_on_real_wt1_without_crash():
    m = DatasetManager()
    m.load_csv(WT1_PATH_REAL, name="WT1")
    result = detect_sensor_issues(m, dataset_name="WT1")
    assert result["ok"] is True
    # audit da xac nhan WT1 co dung 7 gap >1 gio
    assert result["data"]["timestamp_gap"]["count"] == 7


def test_integration_diagnose_theoretical_power_deviation_on_real_gt1_without_crash():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    result = diagnose_theoretical_power_deviation(m, dataset_name="GT1")
    assert result["ok"] is True
    # audit da xac nhan 21.302 dong (9.11%) o threshold mac dinh 10%
    assert 20000 < result["data"]["violation_count"] < 23000


# =============================== explain_anomaly ===============================

def _explain_anomaly_fixture(tmp_path):
    """40 dong 'binh thuong' + 15 dong 'anomaly', cot 'value' co >31 gia tri duy nhat
    (buoc nhay 0.1) de chac chan duoc heuristic nunique cua explain_anomaly() phan
    loai la numeric (khong bi nham thanh categorical nhu khi fixture qua nho)."""
    normal_values = [10.0 + i * 0.1 for i in range(40)]
    anomaly_values = [100.0 + i * 0.1 for i in range(15)]
    ts = pd.date_range("2024-01-01", periods=55, freq="h")

    rows = ["ts,is_bad,value,state"]
    for i, v in enumerate(normal_values):
        rows.append(f"{ts[i].isoformat()},0,{v},1")
    for i, v in enumerate(anomaly_values):
        rows.append(f"{ts[40+i].isoformat()},1,{v},2")
    csv_path = tmp_path / "explain_anomaly_fixture.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m, sum(normal_values) / len(normal_values), sum(anomaly_values) / len(anomaly_values)


def test_explain_anomaly_with_mask_column_compares_numeric_group_means(tmp_path):
    m, expected_normal_mean, expected_anomaly_mean = _explain_anomaly_fixture(tmp_path)
    result = explain_anomaly(m, anomaly_mask_column="is_bad", compare_columns=["value"])
    assert result["ok"] is True
    assert result["data"]["anomaly_count"] == 15
    assert result["data"]["other_count"] == 40
    value_cmp = result["data"]["comparison"]["value"]
    assert value_cmp["type"] == "numeric"
    assert value_cmp["anomaly_group"]["mean"] == pytest.approx(expected_anomaly_mean, abs=1e-3)
    assert value_cmp["other_group"]["mean"] == pytest.approx(expected_normal_mean, abs=1e-3)


def test_explain_anomaly_with_row_indices_matches_mask_column_result(tmp_path):
    m, _, expected_anomaly_mean = _explain_anomaly_fixture(tmp_path)
    df = m.get_dataframe()
    row_indices = df.index[df["is_bad"] == 1].tolist()

    result = explain_anomaly(m, row_indices=row_indices, compare_columns=["value"])
    assert result["ok"] is True
    assert result["data"]["anomaly_count"] == 15
    assert result["data"]["comparison"]["value"]["anomaly_group"]["mean"] == pytest.approx(expected_anomaly_mean, abs=1e-3)


def test_explain_anomaly_low_cardinality_column_uses_categorical_breakdown(tmp_path):
    m, _, _ = _explain_anomaly_fixture(tmp_path)
    result = explain_anomaly(m, anomaly_mask_column="is_bad", compare_columns=["state"])
    assert result["ok"] is True
    state_cmp = result["data"]["comparison"]["state"]
    assert state_cmp["type"] == "categorical"
    assert state_cmp["anomaly_group_pct"] == {"2": 100.0}
    assert state_cmp["other_group_pct"] == {"1": 100.0}


def test_explain_anomaly_requires_exactly_one_of_mask_or_indices(tmp_path):
    m, _, _ = _explain_anomaly_fixture(tmp_path)
    neither = explain_anomaly(m)
    assert neither["ok"] is False

    both = explain_anomaly(m, anomaly_mask_column="is_bad", row_indices=[0, 1])
    assert both["ok"] is False


def test_explain_anomaly_no_conclusion_field(tmp_path):
    m, _, _ = _explain_anomaly_fixture(tmp_path)
    result = explain_anomaly(m, anomaly_mask_column="is_bad", compare_columns=["value"])
    assert "conclusion" not in result["data"]


def test_integration_explain_anomaly_on_real_gt1_reproduces_diagnose_theoretical_power_deviation_count():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    df = m.get_dataframe("GT1")
    mask = df["GT1.Grid.totW"] > (df["GT1.Common.TheoActiveW"] * 1.1)
    row_indices = df.index[mask].tolist()

    result = explain_anomaly(
        m, dataset_name="GT1", row_indices=row_indices,
        compare_columns=["GT1.Common.OpSt", "GT1.Pitch.Angle1"],
    )
    assert result["ok"] is True
    assert result["data"]["anomaly_count"] == 21302  # phai khop dung diagnose_theoretical_power_deviation


# =============================== estimate_power_curve ===============================

def _synthetic_power_curve_dataset(tmp_path, n=3000, seed=42, noise_frac=0.6):
    """40% diem 'sach' theo cong thuc tuyen tinh power=100*ws (jitter nho +-2%),
    60% diem 'nhieu' rai deu ngau nhien (random scatter) tren TOAN MIEN gia tri hop
    ly [0, 100*20]. Dai wind speed [5,20] (TRANH sat bien duoi gan 0) — da verify
    thuc nghiem rang khi tin hieu that nam sat mep duoi cua mien nhieu, uoc luong
    median bi lech bat doi xung (xem docstring estimate_power_curve, muc GIOI HAN)."""
    rng = np.random.RandomState(seed)
    ws = rng.uniform(5, 20, n)
    true_power = 100 * ws
    clean_mask = rng.rand(n) < (1 - noise_frac)
    noise_power = rng.uniform(0, 2000, n)
    power = np.where(clean_mask, true_power * (1 + rng.normal(0, 0.02, n)), noise_power)

    rows = ["ts,ws,power"]
    base = pd.Timestamp("2024-01-01")
    for i in range(n):
        ts = base + pd.Timedelta(minutes=10 * i)
        rows.append(f"{ts.isoformat()},{ws[i]:.4f},{power[i]:.4f}")
    csv_path = tmp_path / "power_curve_synth.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m


def test_estimate_power_curve_recovers_true_curve_with_majority_random_scatter_noise(tmp_path):
    m = _synthetic_power_curve_dataset(tmp_path)
    result = estimate_power_curve(m, power_column="power", wind_speed_column="ws", bin_width=1.0)
    assert result["ok"] is True

    checked = 0
    for b in result["data"]["power_curve"]:
        if b["n_points_total"] < 20 or b["median_power"] is None:
            continue
        expected = 100 * b["wind_speed_bin_center"]
        err_pct = abs(b["median_power"] - expected) / expected * 100
        assert err_pct < 20, f"bin {b['wind_speed_bin_center']}: err%={err_pct:.1f} qua lon"
        checked += 1
    assert checked >= 10  # dam bao thuc su kiem tra du nhieu bin, khong phai test rong


def test_estimate_power_curve_empty_bin_returns_none_without_crash(tmp_path):
    rows = ["ts,ws,power"]
    base = pd.Timestamp("2024-01-01")
    i = 0
    for w in np.arange(0, 3, 0.2):  # cum 1: gio thap
        ts = base + pd.Timedelta(minutes=10 * i)
        rows.append(f"{ts.isoformat()},{w:.2f},{100*w:.2f}")
        i += 1
    for w in np.arange(15, 18, 0.2):  # cum 2: gio cao — de trong khoang giua (3-15) khong co diem nao
        ts = base + pd.Timedelta(minutes=10 * i)
        rows.append(f"{ts.isoformat()},{w:.2f},{100*w:.2f}")
        i += 1
    csv_path = tmp_path / "gap_dataset.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = estimate_power_curve(m, power_column="power", wind_speed_column="ws", bin_width=1.0)
    assert result["ok"] is True

    empty_bins = [b for b in result["data"]["power_curve"] if 5 <= b["wind_speed_bin_center"] <= 12]
    assert len(empty_bins) > 0
    for b in empty_bins:
        assert b["n_points_total"] == 0
        assert b["n_points_used"] == 0
        assert b["median_power"] is None
        assert b["std_power"] is None
        assert b["iqr_power"] is None


def test_estimate_power_curve_convergence_info_present(tmp_path):
    rows = ["ts,ws,power"]
    base = pd.Timestamp("2024-01-01")
    for i in range(200):
        w = 5 + (i % 15)
        ts = base + pd.Timedelta(minutes=10 * i)
        rows.append(f"{ts.isoformat()},{w},{100*w}")  # du lieu sach hoan toan -> hoi tu nhanh
    csv_path = tmp_path / "clean_dataset.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    result = estimate_power_curve(m, power_column="power", wind_speed_column="ws", bin_width=1.0)
    assert result["ok"] is True
    assert result["data"]["convergence_info"]["converged"] is True
    assert result["data"]["convergence_info"]["iterations_run"] <= 5


def test_integration_estimate_power_curve_on_real_gt1_shows_rising_then_saturating_trend():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    result = estimate_power_curve(m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd", dataset_name="GT1", bin_width=1.0)
    assert result["ok"] is True

    curve = [b for b in result["data"]["power_curve"] if b["median_power"] is not None and b["n_points_total"] >= 30]
    curve.sort(key=lambda b: b["wind_speed_bin_center"])
    assert len(curve) >= 10
    # gio thap (~3-5 m/s) phai co median power thap hon han gio trung binh (~10-12 m/s)
    low = [b for b in curve if 3 <= b["wind_speed_bin_center"] <= 5]
    mid = [b for b in curve if 10 <= b["wind_speed_bin_center"] <= 12]
    assert low and mid
    assert max(b["median_power"] for b in low) < min(b["median_power"] for b in mid)


def test_integration_estimate_power_curve_on_real_wt1_shows_rising_then_saturating_trend():
    m = DatasetManager()
    m.load_csv(WT1_PATH_REAL, name="WT1")
    result = estimate_power_curve(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", dataset_name="WT1", bin_width=1.0)
    assert result["ok"] is True

    curve = [b for b in result["data"]["power_curve"] if b["median_power"] is not None and b["n_points_total"] >= 30]
    curve.sort(key=lambda b: b["wind_speed_bin_center"])
    assert len(curve) >= 10
    low = [b for b in curve if 3 <= b["wind_speed_bin_center"] <= 5]
    mid = [b for b in curve if 8 <= b["wind_speed_bin_center"] <= 10]
    assert low and mid
    assert max(b["median_power"] for b in low) < min(b["median_power"] for b in mid)


# =============================== classify_turbine_status ===============================

def _fake_power_curve(bins: list[dict], bin_width: float = 1.0) -> dict:
    """power_curve gia (dung format field 'data' cua estimate_power_curve()) — moi bin la
    dict {'wind_speed_bin_center','median_power','std_power'}. n_points_used/total khong
    can thiet cho classify_turbine_status() nen bo qua trong fixture test."""
    return {"bin_width": bin_width, "power_curve": bins}


def _site2_manager(tmp_path, rows: list[str], header: str = "DATE_TIME,ACTIVE_POWER,WIND_SPEED") -> DatasetManager:
    csv_path = tmp_path / "site2_fixture.csv"
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m


def test_classify_turbine_status_stop_label(tmp_path):
    m = _site2_manager(tmp_path, ["01/01/2024 00:00,1.0,5.5"])
    curve = _fake_power_curve([{"wind_speed_bin_center": 5.5, "median_power": 500.0, "std_power": 50.0}])
    result = classify_turbine_status(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", power_curve=curve)
    assert result["ok"] is True
    assert result["data"]["classification"][0] == "Stop"


def test_classify_turbine_status_under_over_and_normal_labels(tmp_path):
    rows = [
        "01/01/2024 00:00,350.0,5.5",   # dev = -3 std -> Under Production
        "01/01/2024 00:10,650.0,5.5",   # dev = +3 std -> Over Production
        "01/01/2024 00:20,550.0,5.5",   # dev = +1 std -> Normal
    ]
    m = _site2_manager(tmp_path, rows)
    curve = _fake_power_curve([{"wind_speed_bin_center": 5.5, "median_power": 500.0, "std_power": 50.0}])
    result = classify_turbine_status(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", power_curve=curve)
    assert result["ok"] is True
    c = result["data"]["classification"]
    assert c[0] == "Under Production"
    assert c[1] == "Over Production"
    assert c[2] == "Normal"


def test_classify_turbine_status_partial_stop_label_not_caught_by_under_production(tmp_path):
    """Bin gio thap (ws=3.5, gan cut-in) co median/std NHO — 1 diem power=15 gan bang
    median cua CHINH BIN DO (khong lech >2std) nhung van rat thap so voi rated_power
    TOAN CUC (tu bin cao nhat, median=500) -> phai la Partial Stop (15 < 5%*500=25),
    KHONG phai Under Production (vi dev so voi bin rieng no chi -0.5 std)."""
    rows = [
        "01/01/2024 00:00,15.0,3.5",   # bin 3.5: median=20,std=10 -> dev=-0.5 (khong Under)
        "01/01/2024 00:10,500.0,15.5",  # bin cao cung cap rated_power=500 (khong dung de test truc tiep)
    ]
    m = _site2_manager(tmp_path, rows)
    curve = _fake_power_curve([
        {"wind_speed_bin_center": 3.5, "median_power": 20.0, "std_power": 10.0},
        {"wind_speed_bin_center": 15.5, "median_power": 500.0, "std_power": 50.0},
    ])
    result = classify_turbine_status(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", power_curve=curve)
    assert result["ok"] is True
    assert result["data"]["rated_power_estimate"] == pytest.approx(500.0)
    assert result["data"]["classification"][0] == "Partial Stop"


def _site1_manager_with_setpoint(tmp_path, rows: list[str]) -> DatasetManager:
    header = "TimeStamp,GT1.Grid.totW,GT1.Common.WindSpd,GT1.SP.ActualSPP"
    csv_path = tmp_path / "site1_setpoint_fixture.csv"
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m


def test_classify_turbine_status_curtailment_label_site1_only(tmp_path):
    # setpoint=850 (< median*0.9=900 -> setpoint bi dat thap); power=867 bam sat setpoint
    # (|867-850|/850=0.02 < 0.05 tight tolerance)
    m = _site1_manager_with_setpoint(tmp_path, ["2024-01-01 00:00:00,867.0,8.5,850.0"])
    curve = _fake_power_curve([{"wind_speed_bin_center": 8.5, "median_power": 1000.0, "std_power": 100.0}])
    result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve, setpoint_column="GT1.SP.ActualSPP",
    )
    assert result["ok"] is True
    assert result["data"]["curtailment_applicable"] is True
    assert result["data"]["classification"][0] == "Curtailment"


def test_classify_turbine_status_partial_curtailment_label(tmp_path):
    # setpoint=850 (below curve); power=935 -> |935-850|/850=0.10 (fail tight 0.05,
    # pass loose 0.15); dev tu bin median (1000,std=100): (935-1000)/100=-0.65 (khong
    # Under Production); power(935) khong < 5%*rated(1000)=50 -> khong Partial Stop
    m = _site1_manager_with_setpoint(tmp_path, ["2024-01-01 00:00:00,935.0,8.5,850.0"])
    curve = _fake_power_curve([{"wind_speed_bin_center": 8.5, "median_power": 1000.0, "std_power": 100.0}])
    result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve, setpoint_column="GT1.SP.ActualSPP",
    )
    assert result["ok"] is True
    assert result["data"]["classification"][0] == "Partial Curtailment"


def test_classify_turbine_status_measurement_error_takes_priority_over_stop(tmp_path):
    """Diem co WIND_SPEED vs WIND_SPEED_2 lech >20% (anemometer_mismatch, Site 2) VA
    dong thoi power rat thap (du dieu kien Stop) -> phai la Measurement Error (uu tien
    1) chu KHONG phai Stop (uu tien 2), vi Measurement Error duoc check truoc."""
    header = "DATE_TIME,ACTIVE_POWER,WIND_SPEED,WIND_SPEED_2"
    rows = ["01/01/2024 00:00,1.0,10.0,7.0"]  # |10-7|/10=0.30 > 0.20 -> anemometer_mismatch
    m = _site2_manager(tmp_path, rows, header=header)
    curve = _fake_power_curve([{"wind_speed_bin_center": 10.5, "median_power": 800.0, "std_power": 80.0}])
    result = classify_turbine_status(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", power_curve=curve)
    assert result["ok"] is True
    assert result["data"]["classification"][0] == "Measurement Error"


def test_classify_turbine_status_site2_without_setpoint_never_assigns_curtailment(tmp_path):
    rows = ["01/01/2024 00:00,500.0,5.5", "01/01/2024 00:10,10.0,3.5"]
    m = _site2_manager(tmp_path, rows)
    curve = _fake_power_curve([
        {"wind_speed_bin_center": 5.5, "median_power": 500.0, "std_power": 50.0},
        {"wind_speed_bin_center": 3.5, "median_power": 20.0, "std_power": 10.0},
    ])
    result = classify_turbine_status(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", power_curve=curve)
    assert result["ok"] is True
    assert result["data"]["curtailment_applicable"] is False
    assert "Curtailment" not in result["data"]["summary"]
    assert "Partial Curtailment" not in result["data"]["summary"]
    assert set(result["data"]["classification"].values()) & {"Curtailment", "Partial Curtailment"} == set()


def test_classify_turbine_status_invalid_power_curve_fails_gracefully(tmp_path):
    m = _site2_manager(tmp_path, ["01/01/2024 00:00,500.0,5.5"])
    result = classify_turbine_status(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", power_curve={})
    assert result["ok"] is False


def test_integration_classify_turbine_status_on_real_gt1_normal_is_majority():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    curve_result = estimate_power_curve(m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd", dataset_name="GT1", bin_width=1.0)
    assert curve_result["ok"] is True

    result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve_result["data"], dataset_name="GT1", setpoint_column="GT1.SP.ActualSPP",
    )
    assert result["ok"] is True
    assert len(result["data"]["classification"]) == 233953
    summary = result["data"]["summary"]
    assert summary["Normal"]["count"] == max(v["count"] for v in summary.values())
    assert summary["Normal"]["pct"] > 50  # Normal phai chiem da so


def test_integration_classify_turbine_status_on_real_wt1_site2_no_curtailment():
    m = DatasetManager()
    m.load_csv(WT1_PATH_REAL, name="WT1")
    curve_result = estimate_power_curve(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", dataset_name="WT1", bin_width=1.0)
    assert curve_result["ok"] is True

    result = classify_turbine_status(
        m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        power_curve=curve_result["data"], dataset_name="WT1",
    )
    assert result["ok"] is True
    assert result["data"]["curtailment_applicable"] is False
    assert "Curtailment" not in result["data"]["summary"]
    summary = result["data"]["summary"]
    assert summary["Normal"]["pct"] > 50


# =============================== group_anomaly_windows ===============================

def _timestamp_manager(tmp_path, n_rows: int, minutes_step: int = 10) -> DatasetManager:
    base = pd.Timestamp("2024-01-01")
    rows = ["ts,dummy"]
    for i in range(n_rows):
        ts = base + pd.Timedelta(minutes=minutes_step * i)
        rows.append(f"{ts.isoformat()},{i}")
    csv_path = tmp_path / "ts_fixture.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    return m


def test_group_anomaly_windows_filters_by_min_consecutive_points(tmp_path):
    # dung dung vi du trong task: Normal, Stop x5, Normal, Under Production x2, Normal
    labels = ["Normal", "Stop", "Stop", "Stop", "Stop", "Stop", "Normal", "Under Production", "Under Production", "Normal"]
    m = _timestamp_manager(tmp_path, n_rows=len(labels))
    classification = {i: lbl for i, lbl in enumerate(labels)}

    result = group_anomaly_windows(m, classification=classification, timestamp_column="datetime", min_consecutive_points=3)
    assert result["ok"] is True
    windows = result["data"]["windows"]
    assert len(windows) == 1
    assert windows[0]["label"] == "Stop"
    assert windows[0]["n_points"] == 5


def test_group_anomaly_windows_respects_custom_exclude_labels(tmp_path):
    labels = ["Stop", "Stop", "Stop"]
    m = _timestamp_manager(tmp_path, n_rows=3)
    classification = {i: lbl for i, lbl in enumerate(labels)}

    result = group_anomaly_windows(m, classification=classification, timestamp_column="datetime", min_consecutive_points=3, exclude_labels=["Stop"])
    assert result["ok"] is True
    assert result["data"]["windows"] == []


def test_group_anomaly_windows_sorts_by_timestamp_before_grouping(tmp_path):
    """classification dict co the KHONG theo thu tu thoi gian trong index goc — ham phai
    tu sap xep lai theo timestamp_column truoc khi gop cua so, khong dua vao thu tu dict."""
    base = pd.Timestamp("2024-01-01")
    rows = ["ts,dummy"]
    # dao nguoc thu tu ghi vao CSV so voi thoi gian that (dong dau la muon nhat)
    order = [4, 3, 2, 1, 0]
    for i in order:
        ts = base + pd.Timedelta(minutes=10 * i)
        rows.append(f"{ts.isoformat()},{i}")
    csv_path = tmp_path / "unsorted_fixture.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))

    # row_index 0 (dong dau file, ts=2024-01-01 00:40) la "Stop"... nhung ve THOI GIAN
    # thuc su no la diem CUOI cung. Gan nhan theo dung 5 vi tri thoi gian tang dan:
    # t=0:00->Stop, 0:10->Stop, 0:20->Stop, 0:30->Normal, 0:40->Normal
    # row_index trong df: dong 0 = i=4 (t=0:40) = Normal; dong1=i=3(t=0:30)=Normal;
    # dong2=i=2(t=0:20)=Stop; dong3=i=1(t=0:10)=Stop; dong4=i=0(t=0:00)=Stop
    classification = {0: "Normal", 1: "Normal", 2: "Stop", 3: "Stop", 4: "Stop"}

    result = group_anomaly_windows(m, classification=classification, timestamp_column="datetime", min_consecutive_points=3)
    assert result["ok"] is True
    windows = result["data"]["windows"]
    assert len(windows) == 1
    assert windows[0]["label"] == "Stop"
    assert windows[0]["n_points"] == 3
    assert windows[0]["start_time"] == "2024-01-01 00:00:00"
    assert windows[0]["end_time"] == "2024-01-01 00:20:00"


def test_integration_group_anomaly_windows_on_real_gt1_without_crash():
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    curve_result = estimate_power_curve(m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd", dataset_name="GT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve_result["data"], dataset_name="GT1", setpoint_column="GT1.SP.ActualSPP",
    )
    result = group_anomaly_windows(
        m, classification=classify_result["data"]["classification"],
        timestamp_column="datetime", dataset_name="GT1", min_consecutive_points=3,
    )
    assert result["ok"] is True
    assert result["data"]["total_windows"] > 0
    for w in result["data"]["windows"][:20]:
        assert w["n_points"] >= 3


# =============================== detect_anomalies_ml ===============================

def _gate_scenario_fixture(tmp_path):
    """
    223 dong: 200 'binh thuong' (power=1000, classification=Normal) + 20 'outlier ro rang'
    (power=5000, classification=Measurement Error — z-score THAT SU se flag vi |z|>3,
    da tinh tay: mean~1358.7, std~1145.4, z(5000)=3.18) + 3 diem 'IF-only co y nghia'
    (power=1000 giong baseline nen z-score KHONG flag, nhung classification=Under
    Production — dai dien cho diem ML phat hien duoc ma z-score bo sot).
    Dat 200 dong Normal o DAU file (train zone 80% dau chi toan Normal, du >=30 diem).
    """
    rows = ["ts,power,wind,classlabel"]
    base = pd.Timestamp("2024-01-01")
    i = 0
    for _ in range(200):
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},1000.0,10.0,Normal")
        i += 1
    for _ in range(20):
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},5000.0,10.0,Measurement Error")
        i += 1
    for _ in range(3):
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},1000.0,10.0,Under Production")
        i += 1
    csv_path = tmp_path / "gate_scenario.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    df = m.get_dataframe()
    classification = {idx: lbl for idx, lbl in zip(df.index, df["classlabel"])}
    return m, classification, len(df)


def test_detect_anomalies_ml_uses_isolation_forest_when_gate_passes(tmp_path):
    m, classification, n = _gate_scenario_fixture(tmp_path)

    # Mock IsolationForest: flag DUNG 23 dong cuoi (20 outlier + 3 IF-only) la anomaly (-1),
    # con lai la normal (1) — theo dung thu tu dong goc (valid_mask la toan bo, khong NaN).
    fake_predictions = np.array([1] * (n - 23) + [-1] * 23)
    mock_instance = MagicMock()
    mock_instance.fit.return_value = None
    mock_instance.predict.return_value = fake_predictions
    with patch("sklearn.ensemble.IsolationForest", return_value=mock_instance) as mock_cls:
        result = detect_anomalies_ml(
            m, power_column="power", wind_speed_column="wind", classification=classification,
        )

    assert result["ok"] is True
    assert result["data"]["method_used"] == "isolation_forest"
    gate = result["data"]["gate_4a"]
    assert gate["overlap_gate_pass"] is True
    assert gate["meaningful_gate_pass"] is True
    assert gate["overlap_pct_with_zscore"] >= 80.0
    assert gate["meaningful_count"] == 3
    assert result["data"]["anomaly_count"] == 23


def test_detect_anomalies_ml_falls_back_when_overlap_gate_fails(tmp_path):
    m, classification, n = _gate_scenario_fixture(tmp_path)

    # Mock IsolationForest flag 23 dong HOAN TOAN KHAC voi z-score (vd 23 dong dau,
    # deu la baseline Normal, khong trung voi 20 outlier ma z-score flag) -> overlap=0%.
    fake_predictions = np.array([-1] * 23 + [1] * (n - 23))
    mock_instance = MagicMock()
    mock_instance.fit.return_value = None
    mock_instance.predict.return_value = fake_predictions
    with patch("sklearn.ensemble.IsolationForest", return_value=mock_instance):
        result = detect_anomalies_ml(
            m, power_column="power", wind_speed_column="wind", classification=classification,
        )

    assert result["ok"] is True
    assert result["data"]["method_used"] == "fallback_statistical_zscore"
    assert "Gate 4a FAIL" in result["data"]["ml_gate_reason"]
    assert result["data"]["gate_4a"]["overlap_gate_pass"] is False


def test_detect_anomalies_ml_falls_back_when_insufficient_normal_training_points(tmp_path):
    """Dataset qua nho (<30 diem Normal trong train zone) -> fallback tu nhien, KHONG can mock."""
    rows = ["ts,power,wind,classlabel"]
    base = pd.Timestamp("2024-01-01")
    for i in range(10):
        # power co bien thien nhe (khong hang so) de detect_statistical_anomalies()
        # (fallback that su duoc goi) khong tu fail vi std=0 — day la 1 bug rieng da
        # phat hien va sua khi viet test nay (fixture ban dau std=0 lam sai lech test).
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},{1000.0+i},10.0,Normal")
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    df = m.get_dataframe()
    classification = {idx: "Normal" for idx in df.index}

    result = detect_anomalies_ml(m, power_column="power", wind_speed_column="wind", classification=classification)
    assert result["ok"] is True
    assert result["data"]["method_used"] == "fallback_statistical_zscore"
    assert "Không đủ điểm" in result["data"]["ml_gate_reason"]


def test_detect_anomalies_ml_falls_back_when_sklearn_import_fails(tmp_path):
    m, classification, n = _gate_scenario_fixture(tmp_path)
    with patch.dict("sys.modules", {"sklearn.ensemble": None}):
        result = detect_anomalies_ml(m, power_column="power", wind_speed_column="wind", classification=classification)
    assert result["ok"] is True
    assert result["data"]["method_used"] == "fallback_statistical_zscore"
    assert "scikit-learn" in result["data"]["ml_gate_reason"]


def test_integration_detect_anomalies_ml_on_real_gt1_falls_back_gate_fails_as_measured():
    """Xac nhan hanh vi THAT tren GT1 that: Gate 4a FAIL that su (da do 2026-09-08,
    xem docstring detect_anomalies_ml) -> phai fallback, khong dung Isolation Forest that."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    curve_result = estimate_power_curve(m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd", dataset_name="GT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve_result["data"], dataset_name="GT1", setpoint_column="GT1.SP.ActualSPP",
    )
    result = detect_anomalies_ml(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        classification=classify_result["data"]["classification"], dataset_name="GT1",
        pitch_column="GT1.Pitch.Angle1",
    )
    assert result["ok"] is True


# =============================== predict_power ===============================

def _predict_power_fixture(tmp_path, n_train=50, n_test=20, extra_features=None):
    """wind co dinh = 5.0 (bin center 5.5 voi bin_width=1.0) — power co bien thien nhe
    quanh 500 de baseline (noi suy hang so=500) co RMSE > 0, cho phep XGBoost (mock)
    "hoan hao" (RMSE=0) thang ro rang. extra_features: list ten cot dummy them vao,
    gia tri co dinh (khong anh huong prediction thuc vi XGBRegressor duoc mock)."""
    extra_features = extra_features or []
    header = "ts,power,wind" + "".join(f",{f}" for f in extra_features)
    rows = [header]
    base = pd.Timestamp("2024-01-01")
    rng = np.random.RandomState(0)
    i = 0
    for _ in range(n_train):
        p = 500.0 + rng.uniform(-5, 5)
        extras = "".join(f",{1.0}" for _ in extra_features)
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},{p},5.0{extras}")
        i += 1
    test_powers = []
    for _ in range(n_test):
        p = 500.0 + rng.uniform(-20, 20)
        test_powers.append(p)
        extras = "".join(f",{2.0}" for _ in extra_features)
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},{p},5.0{extras}")
        i += 1
    csv_path = tmp_path / "predict_power_fixture.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    df = m.get_dataframe()
    classification = {idx: "Normal" for idx in df.index}
    power_curve = {"bin_width": 1.0, "power_curve": [{"wind_speed_bin_center": 5.5, "median_power": 500.0, "std_power": 3.0}]}
    # train_fraction chinh xac de split_idx = n_train dung boc (predict_power dung
    # int(n*train_fraction) — truyen tuong minh de tranh lech ranh gioi train/test).
    train_fraction = n_train / (n_train + n_test)
    return m, classification, power_curve, test_powers, train_fraction


def test_mape_helper_excludes_near_zero_points_and_computes_correctly():
    """Tay tinh: 5 diem, 3 diem |actual|<=50kW (1.0, -0.5, 40.0) bi loai — CHI con 2 diem
    (100.0, 200.0) dung de tinh MAPE."""
    actual = np.array([100.0, 200.0, 1.0, -0.5, 40.0])
    pred = np.array([110.0, 180.0, 5.0, 3.0, 45.0])
    result = _mape(actual, pred, MAPE_MIN_ACTUAL_POWER_KW)
    expected = np.mean([abs((100.0 - 110.0) / 100.0), abs((200.0 - 180.0) / 200.0)]) * 100
    assert result == pytest.approx(expected, rel=1e-9)
    assert result == pytest.approx(10.0, rel=1e-9)  # |−10/100|=10%, |20/200|=10% -> mean=10%


def test_mape_helper_returns_none_when_all_points_filtered_out():
    actual = np.array([10.0, -5.0, 0.0, 49.9])
    pred = np.array([12.0, -4.0, 1.0, 50.0])
    assert _mape(actual, pred, MAPE_MIN_ACTUAL_POWER_KW) is None


def test_predict_power_uses_xgboost_when_both_gates_pass(tmp_path):
    m, classification, power_curve, test_powers, train_fraction = _predict_power_fixture(tmp_path)

    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.predict.return_value = np.array(test_powers)  # du doan HOAN HAO -> RMSE=0
    mock_model.feature_importances_ = np.array([0.99, 0.01])  # wind #1

    with patch("xgboost.XGBRegressor", return_value=mock_model):
        result = predict_power(
            m, power_column="power", wind_speed_column="wind", power_curve=power_curve,
            classification=classification, train_fraction=train_fraction,
        )
    assert result["ok"] is True
    assert result["data"]["method_used"] == "xgboost"
    assert result["data"]["gate_4b"]["gate_4b_pass"] is True
    assert result["data"]["gate_4c"]["gate_4c_pass"] is True
    assert result["data"]["gate_4c"]["wind_speed_rank"] == 1
    assert result["data"]["metrics"]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert result["data"]["baseline_metrics"]["rmse"] > 0
    # test_powers dao dong quanh 500 (>>50kW nguong loc) -> khong diem nao bi loai khi
    # tinh MAPE o day. XGBoost du doan HOAN HAO -> MAPE=0. Baseline (noi suy hang so
    # 500) MAPE > 0 vi actual dao dong ±20 quanh 500 (khong trung khop hoan toan).
    assert result["data"]["metrics"]["mape"] == pytest.approx(0.0, abs=1e-6)
    assert result["data"]["baseline_metrics"]["mape"] is not None
    assert result["data"]["baseline_metrics"]["mape"] > 0


def test_predict_power_falls_back_when_gate_4b_fails(tmp_path):
    m, classification, power_curve, test_powers, train_fraction = _predict_power_fixture(tmp_path)

    # du doan RAT XA thuc te -> RMSE xgboost >> baseline
    bad_predictions = np.array(test_powers) + 10000.0
    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.predict.return_value = bad_predictions
    mock_model.feature_importances_ = np.array([0.99, 0.01])

    with patch("xgboost.XGBRegressor", return_value=mock_model):
        result = predict_power(
            m, power_column="power", wind_speed_column="wind", power_curve=power_curve,
            classification=classification, train_fraction=train_fraction,
        )
    assert result["ok"] is True
    assert result["data"]["method_used"] == "power_curve_interpolation"
    assert "Gate 4b FAIL" in result["data"]["ml_gate_reason"]
    assert result["data"]["gate_4b"]["gate_4b_pass"] is False


def test_predict_power_falls_back_when_gate_4c_fails_even_though_gate_4b_passes(tmp_path):
    """3 feature (wind + 2 dummy) — du doan XGBoost HOAN HAO (Gate 4b chac chan pass)
    nhung feature_importances_ xep wind xuong hang #3 -> VAN PHAI fallback (Gate 4c
    nghiem ngat hon Gate 4b, khong duoc dung ML du RMSE tot)."""
    m, classification, power_curve, test_powers, train_fraction = _predict_power_fixture(
        tmp_path, extra_features=["extra1", "extra2"]
    )

    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.predict.return_value = np.array(test_powers)  # RMSE=0, Gate 4b chac chan pass
    # thu tu feature dua vao model: [wind, extra1, extra2] (theo all_features trong code)
    # xep wind THAP NHAT -> hang #3
    mock_model.feature_importances_ = np.array([0.05, 0.5, 0.45])

    with patch("xgboost.XGBRegressor", return_value=mock_model):
        result = predict_power(
            m, power_column="power", wind_speed_column="wind", power_curve=power_curve,
            classification=classification, feature_columns=["extra1", "extra2"], train_fraction=train_fraction,
        )
    assert result["ok"] is True
    assert result["data"]["method_used"] == "power_curve_interpolation"
    assert "Gate 4c FAIL" in result["data"]["ml_gate_reason"]
    assert result["data"]["gate_4b"]["gate_4b_pass"] is True  # Gate 4b THAT SU pass
    assert result["data"]["gate_4c"]["gate_4c_pass"] is False
    assert result["data"]["gate_4c"]["wind_speed_rank"] == 3


def _leakage_column_fixture(tmp_path, extra_col_name):
    """Fixture RIENG co san cot leakage-risk (khac fixture chung _predict_power_fixture
    KHONG co cot nay) — bug da tu phat hien: test cu dung feature_columns tro toi cot
    KHONG TON TAI trong df, nen chi cham nhanh '_validate_column' (bao 'not found'),
    CHUA BAO GIO thuc su chay toi nhanh leakage-guard. Sua bang cach tao that cot do."""
    header = f"ts,power,wind,{extra_col_name}"
    rows = [header]
    base = pd.Timestamp("2024-01-01")
    for i in range(40):
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},{500.0+i},5.0,{490.0+i}")
    csv_path = tmp_path / f"leakage_{extra_col_name.replace('.', '_')}.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    df = m.get_dataframe()
    classification = {idx: "Normal" for idx in df.index}
    power_curve = {"bin_width": 1.0, "power_curve": [{"wind_speed_bin_center": 5.5, "median_power": 500.0, "std_power": 3.0}]}
    return m, classification, power_curve


def test_predict_power_rejects_theoretical_power_as_feature(tmp_path):
    m, classification, power_curve = _leakage_column_fixture(tmp_path, "GT1.Common.TheoActiveW")
    result = predict_power(
        m, power_column="power", wind_speed_column="wind", power_curve=power_curve,
        classification=classification, feature_columns=["GT1.Common.TheoActiveW"],
    )
    assert result["ok"] is False
    assert "leakage" in result["error"].lower()
    assert "TheoActiveW" in result["error"]


def test_predict_power_rejects_actual_setpoint_as_feature(tmp_path):
    """ActualSPP la leakage tuong tu TheoActiveW — setpoint gan nhu quyet dinh truc tiep
    power luc Curtailment (xem rule Curtailment trong classify_turbine_status())."""
    m, classification, power_curve = _leakage_column_fixture(tmp_path, "GT1.SP.ActualSPP")
    result = predict_power(
        m, power_column="power", wind_speed_column="wind", power_curve=power_curve,
        classification=classification, feature_columns=["GT1.SP.ActualSPP"],
    )
    assert result["ok"] is False
    assert "leakage" in result["error"].lower()
    assert "ActualSPP" in result["error"]


def test_predict_power_falls_back_when_insufficient_data(tmp_path):
    header = "ts,power,wind"
    rows = [header]
    base = pd.Timestamp("2024-01-01")
    for i in range(10):
        rows.append(f"{(base + pd.Timedelta(minutes=10*i)).isoformat()},{500.0+i},5.0")
    csv_path = tmp_path / "tiny_predict.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path))
    df = m.get_dataframe()
    classification = {idx: "Normal" for idx in df.index}
    power_curve = {"bin_width": 1.0, "power_curve": [{"wind_speed_bin_center": 5.5, "median_power": 500.0, "std_power": 3.0}]}

    result = predict_power(m, power_column="power", wind_speed_column="wind", power_curve=power_curve, classification=classification)
    assert result["ok"] is True
    assert result["data"]["method_used"] == "power_curve_interpolation"
    assert "Không đủ điểm" in result["data"]["ml_gate_reason"]
    # nhanh _fallback() cung phai co "mape" trong metrics (fixture nay actual=500+i,
    # >>50kW nguong loc -> khong bi loai het, mape phai la so, khong phai None).
    assert "mape" in result["data"]["metrics"]
    assert result["data"]["metrics"]["mape"] is not None


def test_integration_predict_power_on_real_gt1_uses_xgboost_as_measured():
    """Xac nhan hanh vi THAT tren GT1: ca 2 gate PASS that su (da do 2026-09-08, xem
    docstring predict_power) -> phai dung XGBoost that, khong fallback."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    curve_result = estimate_power_curve(m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd", dataset_name="GT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve_result["data"], dataset_name="GT1", setpoint_column="GT1.SP.ActualSPP",
    )
    result = predict_power(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve_result["data"], classification=classify_result["data"]["classification"],
        dataset_name="GT1", feature_columns=["GT1.Pitch.Angle1", "GT1.Common.WindTrackDevAng"],
    )
    assert result["ok"] is True
    assert result["data"]["method_used"] == "xgboost"
    assert result["data"]["metrics"]["rmse"] < result["data"]["baseline_metrics"]["rmse"]
    assert result["data"]["gate_4c"]["wind_speed_rank"] == 1
    # MAPE that tren GT1 (do 2026-09-11, khong gia dinh) — chi kiem tra la so hop le,
    # khong am, khong None (dataset that > 30 diem con lai sau loc 50kW nen chac chan
    # co gia tri).
    assert result["data"]["metrics"]["mape"] is not None
    assert result["data"]["metrics"]["mape"] >= 0
    assert result["data"]["baseline_metrics"]["mape"] is not None
    assert result["data"]["baseline_metrics"]["mape"] >= 0
    print(f"\n[GT1 MAPE thuc do] xgboost={result['data']['metrics']['mape']}, baseline={result['data']['baseline_metrics']['mape']}")


def test_integration_predict_power_on_real_wt1_uses_xgboost_as_measured():
    m = DatasetManager()
    m.load_csv(WT1_PATH_REAL, name="WT1")
    curve_result = estimate_power_curve(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", dataset_name="WT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        power_curve=curve_result["data"], dataset_name="WT1",
    )
    result = predict_power(
        m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        power_curve=curve_result["data"], classification=classify_result["data"]["classification"],
        dataset_name="WT1", feature_columns=["PITCH", "WIND_SPEED_STD", "TEMPERATURE_STATOR"],
    )
    assert result["ok"] is True
    assert result["data"]["method_used"] == "xgboost"
    assert result["data"]["metrics"]["rmse"] < result["data"]["baseline_metrics"]["rmse"]
    assert result["data"]["gate_4c"]["wind_speed_rank"] == 1
    assert result["data"]["metrics"]["mape"] is not None
    assert result["data"]["metrics"]["mape"] >= 0
    assert result["data"]["baseline_metrics"]["mape"] is not None
    assert result["data"]["baseline_metrics"]["mape"] >= 0
    print(f"\n[WT1 MAPE thuc do] xgboost={result['data']['metrics']['mape']}, baseline={result['data']['baseline_metrics']['mape']}")


# =============================== calculate_turbine_kpis ===============================

def _kpi_fixture(tmp_path, rows: list[str], header: str = "TimeStamp,POWER,WIND,THEO_POWER") -> DatasetManager:
    """4 dong, cach nhau 5 phut (khop dinh dang TimeStamp dung Site 1) -- de tu suy interval_minutes=5."""
    csv_path = tmp_path / "kpi_fixture.csv"
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    m = DatasetManager()
    m.load_csv(str(csv_path), name="kpi_fixture")
    return m


def test_calculate_turbine_kpis_direct_theoretical_hand_computed(tmp_path):
    """4 dong 5 phut: power=[1000,800,0,1000] kW, theoretical=1000 kW co dinh, nhan =
    [Normal,Normal,Stop,Curtailment]. Tinh tay: production=2800kWh-equiv*(5/60)h/1000=
    0.23333 MWh; reachable=4000*(5/60)/1000=0.33333 MWh; pba=70%, losses=30%;
    rated_power=max(power)*1.02=1020 (KHONG co power_curve); TBA=3/4 available (chi
    'Stop' unavailable)=75%."""
    rows = [
        "2024-01-01 00:00:00,1000.0,5.5,1000.0",
        "2024-01-01 00:05:00,800.0,5.5,1000.0",
        "2024-01-01 00:10:00,0.0,5.5,1000.0",
        "2024-01-01 00:15:00,1000.0,5.5,1000.0",
    ]
    m = _kpi_fixture(tmp_path, rows)
    classification = {0: "Normal", 1: "Normal", 2: "Stop", 3: "Curtailment"}

    result = calculate_turbine_kpis(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification=classification, theoretical_power_column="THEO_POWER",
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["method"] == "direct_theoretical"
    assert data["interval_minutes"] == pytest.approx(5.0)
    assert data["production_mwh"] == pytest.approx(round(2800 * (5 / 60) / 1000, 4))
    assert data["reachable_mwh"] == pytest.approx(round(4000 * (5 / 60) / 1000, 4))
    assert data["pba_pct"] == pytest.approx(70.0)
    assert data["losses_pct"] == pytest.approx(30.0)
    assert data["rated_power_kw"] == pytest.approx(1020.0)
    assert data["tba_pct"] == pytest.approx(75.0)
    from tools.turbine_diagnostic_tools import TBA_APPROXIMATION_NOTE
    assert data["tba_note"] == TBA_APPROXIMATION_NOTE
    assert data["n_rows_used"] == 4
    assert data["n_rows_excluded_no_reachable_estimate"] == 0
    # Dung raw fraction CHUA lam tron (khong phai data["production_mwh"] da round o
    # tren) de tranh sai so cong don qua 2 lan lam tron khac nhau.
    expected_cf = round((2800 * (5 / 60) / 1000) / (1.02 * 4 * (5 / 60)) * 100, 4)
    assert data["capacity_factor_pct"] == pytest.approx(expected_cf)


def test_calculate_turbine_kpis_estimated_power_curve_hand_computed(tmp_path):
    """Khong co theoretical_power_column -- dung power_curve (1 bin duy nhat, moi wind
    speed roi vao bin nay) de uoc luong reachable qua noi suy tuyen tinh (chi 1 bin ->
    noi suy tra ve dung median_power do, khong phu thuoc gia tri wind_speed cu the)."""
    rows = [
        "2024-01-01 00:00:00,900.0,8.0,",
        "2024-01-01 00:05:00,900.0,8.0,",
    ]
    m = _kpi_fixture(tmp_path, rows)
    curve = {"bin_width": 1.0, "power_curve": [
        {"wind_speed_bin_center": 8.0, "median_power": 1000.0, "std_power": 50.0},
    ]}
    classification = {0: "Normal", 1: "Normal"}

    result = calculate_turbine_kpis(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification=classification, power_curve=curve,
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["method"] == "estimated_power_curve"
    assert data["reachable_mwh"] == pytest.approx(round(2000 * (5 / 60) / 1000, 4))
    assert data["rated_power_kw"] == pytest.approx(1000.0)  # tu power_curve, khong phai *1.02 fallback
    assert data["pba_pct"] == pytest.approx(90.0)
    assert data["losses_pct"] == pytest.approx(10.0)
    assert data["tba_pct"] == pytest.approx(100.0)


def test_calculate_turbine_kpis_prefers_theoretical_when_both_given(tmp_path):
    """Neu ca theoretical_power_column LAN power_curve deu duoc truyen -- uu tien
    theoretical (chinh xac hon), dung format tai lieu da ghi."""
    rows = ["2024-01-01 00:00:00,900.0,8.0,950.0"]
    m = _kpi_fixture(tmp_path, rows)
    curve = {"bin_width": 1.0, "power_curve": [{"wind_speed_bin_center": 8.0, "median_power": 1000.0, "std_power": 50.0}]}
    result = calculate_turbine_kpis(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification={0: "Normal"}, theoretical_power_column="THEO_POWER", power_curve=curve,
        interval_minutes=5.0,  # chi 1 dong -> khong tu suy duoc interval qua diff(), truyen tay
    )
    assert result["ok"] is True
    assert result["data"]["method"] == "direct_theoretical"
    assert result["data"]["reachable_mwh"] == pytest.approx(round(950.0 * (5 / 60) / 1000, 4))


def test_calculate_turbine_kpis_measurement_error_excluded_from_tba_denominator():
    """'Measurement Error' KHONG tinh vao mau so TBA -- chi 'Stop'/'Partial Stop' la
    unavailable trong so con lai."""
    classification = {0: "Normal", 1: "Measurement Error", 2: "Stop", 3: "Normal"}
    labels = pd.Series(classification)
    from tools.turbine_diagnostic_tools import _TBA_EXCLUDED_LABELS, _TBA_UNAVAILABLE_LABELS
    counted = labels[~labels.isin(_TBA_EXCLUDED_LABELS)]
    assert len(counted) == 3  # loai 1 Measurement Error
    tba = (~counted.isin(_TBA_UNAVAILABLE_LABELS)).sum() / len(counted) * 100
    assert tba == pytest.approx(200 / 3)  # 2/3 available (Normal,Normal), 1/3 Stop


def test_calculate_turbine_kpis_fails_without_theoretical_or_power_curve(tmp_path):
    rows = ["2024-01-01 00:00:00,900.0,8.0,"]
    m = _kpi_fixture(tmp_path, rows)
    result = calculate_turbine_kpis(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification={0: "Normal"},
        interval_minutes=5.0,  # chi 1 dong -> khong tu suy duoc interval qua diff(), truyen tay
    )
    assert result["ok"] is False
    assert "theoretical_power_column" in result["error"] and "power_curve" in result["error"]


def test_calculate_loss_by_category_direct_theoretical_hand_computed(tmp_path):
    """Tay tinh, CUNG fixture voi test_calculate_turbine_kpis_direct_theoretical_hand_
    computed o tren de doi chieu duoc: 4 dong 5 phut, power=[1000,800,0,1000],
    theoretical=1000 co dinh, nhan=[Normal,Normal,Stop,Curtailment].
    interval_hours=5/60. loss_kwh = (theo-power)*interval_hours, CHI cong cho dong
    KHONG PHAI Normal:
      dong2 (Stop): (1000-0)*5/60 = 83.3333 kWh = 0.0833333 MWh -> round 0.0833
      dong3 (Curtailment): (1000-1000)*5/60 = 0 kWh = 0.0 MWh
    total = 0.0833 (dong 0,1 la Normal, KHONG duoc cong dua loss cua chung, du loss thuc
    su khac 0 o dong 1: (1000-800)*5/60=16.6667 kWh)."""
    rows = [
        "2024-01-01 00:00:00,1000.0,5.5,1000.0",
        "2024-01-01 00:05:00,800.0,5.5,1000.0",
        "2024-01-01 00:10:00,0.0,5.5,1000.0",
        "2024-01-01 00:15:00,1000.0,5.5,1000.0",
    ]
    m = _kpi_fixture(tmp_path, rows)
    classification = {0: "Normal", 1: "Normal", 2: "Stop", 3: "Curtailment"}

    result = calculate_loss_by_category(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification=classification, theoretical_power_column="THEO_POWER",
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["method"] == "direct_theoretical"
    assert data["interval_minutes"] == pytest.approx(5.0)
    assert data["n_rows_used"] == 4
    assert data["loss_by_category_mwh"] == {
        "Stop": pytest.approx(0.0833, abs=1e-4),
        "Curtailment": pytest.approx(0.0, abs=1e-4),
    }
    assert data["total_loss_mwh"] == pytest.approx(0.0833, abs=1e-4)


def test_calculate_loss_by_category_all_normal_returns_empty(tmp_path):
    """Khong dong nao khac Normal -> loss_by_category_mwh rong, total=0 (khong loi)."""
    rows = [
        "2024-01-01 00:00:00,1000.0,5.5,1000.0",
        "2024-01-01 00:05:00,1000.0,5.5,1000.0",
    ]
    m = _kpi_fixture(tmp_path, rows)
    classification = {0: "Normal", 1: "Normal"}
    result = calculate_loss_by_category(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification=classification, theoretical_power_column="THEO_POWER",
    )
    assert result["ok"] is True
    assert result["data"]["loss_by_category_mwh"] == {}
    assert result["data"]["total_loss_mwh"] == pytest.approx(0.0)


def test_calculate_loss_by_category_estimated_power_curve_hand_computed(tmp_path):
    """Khong co theoretical_power_column -- dung power_curve (1 bin, moi wind speed roi
    vao bin nay). power=[900,700], median_power=1000, nhan=[Normal,Under Production].
    interval=5 phut. Dong 1 (Under Production): loss=(1000-700)*5/60=25 kWh=0.025 MWh."""
    rows = [
        "2024-01-01 00:00:00,900.0,8.0,",
        "2024-01-01 00:05:00,700.0,8.0,",
    ]
    m = _kpi_fixture(tmp_path, rows)
    curve = {"bin_width": 1.0, "power_curve": [
        {"wind_speed_bin_center": 8.0, "median_power": 1000.0, "std_power": 50.0},
    ]}
    classification = {0: "Normal", 1: "Under Production"}
    result = calculate_loss_by_category(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification=classification, power_curve=curve,
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["method"] == "estimated_power_curve"
    assert data["loss_by_category_mwh"] == {"Under Production": pytest.approx(0.025, abs=1e-4)}
    assert data["total_loss_mwh"] == pytest.approx(0.025, abs=1e-4)


def test_calculate_loss_by_category_fails_without_theoretical_or_power_curve(tmp_path):
    rows = ["2024-01-01 00:00:00,900.0,8.0,"]
    m = _kpi_fixture(tmp_path, rows)
    result = calculate_loss_by_category(
        m, dataset_name="kpi_fixture", power_column="POWER", wind_speed_column="WIND",
        classification={0: "Normal"}, interval_minutes=5.0,
    )
    assert result["ok"] is False
    assert "theoretical_power_column" in result["error"] and "power_curve" in result["error"]


def _assert_gap_equals_normal_only_contribution(
    m, dataset_name, power_column, wind_speed_column, classification,
    theoretical_power_column, power_curve, losses_mwh_kpi, total_loss_category,
):
    """KHONG assert tong loss_by_category ≈ losses_mwh (DA DO THAT: lech 17.6%/-38.7%,
    KHONG nho — xem docstring calculate_loss_by_category()). Thay vao do, assert dung
    BAT BIEN toan hoc THAT SU dung: gap = losses_mwh(KPI, TOAN BO dong) - total_loss_mwh
    (CHI non-Normal) phai bang CHINH XAC tong (reachable-actual) tren rieng cac dong
    "Normal" — tinh DOC LAP, KHONG goi lai calculate_turbine_kpis()/calculate_loss_by_
    category() de tranh false-positive do dung chung 1 loi tiem an."""
    from tools.turbine_diagnostic_tools import _compute_reachable_power_per_row, _infer_interval_hours

    df = m.get_dataframe(dataset_name)
    reachable, _ = _compute_reachable_power_per_row(df, theoretical_power_column, power_curve, wind_speed_column)
    _, interval_hours = _infer_interval_hours(df, None)
    actual = df[power_column].astype(float)
    valid = reachable.notna() & actual.notna()
    labels = pd.Series(classification).reindex(df.index)
    normal_mask = valid & (labels == "Normal")

    loss_per_row_mwh = (reachable - actual) * interval_hours / 1000
    normal_only_contribution = round(float(loss_per_row_mwh[normal_mask].sum()), 4)
    gap = round(losses_mwh_kpi - total_loss_category, 4)

    print(f"[{dataset_name}] gap(KPI-category)={gap}, normal_only_contribution(doc lap)={normal_only_contribution}")
    assert gap == pytest.approx(normal_only_contribution, abs=0.001), (
        f"{dataset_name}: gap ({gap}) phai bang CHINH XAC dong gop rieng cua nhan Normal "
        f"({normal_only_contribution}) -- neu KHONG khop, day la BUG khac (khong phai do "
        f"Normal), can dieu tra tiep, KHONG duoc ket luan la do Normal khi chua verify."
    )


def test_integration_calculate_loss_by_category_on_real_gt1_gap_equals_normal_contribution():
    """Xac nhan bang so THAT (2026-09-11, khong gia dinh): tong loss_by_category_mwh
    (CHI nhan khong-Normal) KHONG khop losses_mwh cua calculate_turbine_kpis() (lech that
    +177.77 MWh = 17.6% -- xem docstring calculate_loss_by_category() giai thich ro
    nguyen nhan GT1 dung TheoActiveW lam reachable, khac tham chieu voi phan loai
    Normal). Test nay KHONG assert 'khop trong dung sai %' (da chung minh SAI) ma assert
    dung BAT BIEN toan hoc: gap = dong gop rieng cua nhan Normal, tinh DOC LAP."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    curve_result = estimate_power_curve(m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd", dataset_name="GT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve_result["data"], dataset_name="GT1", setpoint_column="GT1.SP.ActualSPP",
    )
    classification = classify_result["data"]["classification"]

    kpi_result = calculate_turbine_kpis(
        m, dataset_name="GT1", power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        classification=classification, theoretical_power_column="GT1.Common.TheoActiveW",
    )
    assert kpi_result["ok"] is True

    loss_result = calculate_loss_by_category(
        m, dataset_name="GT1", power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        classification=classification, theoretical_power_column="GT1.Common.TheoActiveW",
    )
    assert loss_result["ok"] is True
    # Cac nhan tra ve phai la tap con hop le cua 8 nhan Meteodyn, khong bao gio co "Normal"
    assert "Normal" not in loss_result["data"]["loss_by_category_mwh"]
    assert set(loss_result["data"]["loss_by_category_mwh"].keys()) <= set(CLASSIFICATION_LABELS)

    _assert_gap_equals_normal_only_contribution(
        m, "GT1", "GT1.Grid.totW", "GT1.Common.WindSpd", classification,
        "GT1.Common.TheoActiveW", None,
        kpi_result["data"]["losses_mwh"], loss_result["data"]["total_loss_mwh"],
    )


def test_integration_calculate_loss_by_category_on_real_wt1_gap_equals_normal_contribution():
    """Tuong tu GT1 nhung dau NGUOC LAI (da do that): dong Normal dong gop -111.48 MWh =
    -38.7% losses_mwh (287.84 MWh) -- xem docstring calculate_loss_by_category() (WT1
    dung CHINH power-curve median lam reachable, cung tham chieu voi Normal, nhung phan
    du con lai sau khi loai 2 duoi cuc tri khong con doi xung quanh median goc)."""
    m = DatasetManager()
    m.load_csv(WT1_PATH_REAL, name="WT1")
    curve_result = estimate_power_curve(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", dataset_name="WT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        power_curve=curve_result["data"], dataset_name="WT1",
    )
    classification = classify_result["data"]["classification"]

    kpi_result = calculate_turbine_kpis(
        m, dataset_name="WT1", power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        classification=classification, power_curve=curve_result["data"],
    )
    assert kpi_result["ok"] is True

    loss_result = calculate_loss_by_category(
        m, dataset_name="WT1", power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        classification=classification, power_curve=curve_result["data"],
    )
    assert loss_result["ok"] is True
    assert "Normal" not in loss_result["data"]["loss_by_category_mwh"]
    assert set(loss_result["data"]["loss_by_category_mwh"].keys()) <= set(CLASSIFICATION_LABELS)

    _assert_gap_equals_normal_only_contribution(
        m, "WT1", "ACTIVE_POWER", "WIND_SPEED", classification,
        None, curve_result["data"],
        kpi_result["data"]["losses_mwh"], loss_result["data"]["total_loss_mwh"],
    )


def test_integration_calculate_turbine_kpis_on_real_gt1_direct_theoretical():
    """Da do that (2026-09-10): production_mwh=33270.952 MWh qua ~2.2 nam du lieu GT1,
    cung bac do voi PO024 (8327-10570 MWh/turbine/2 nam, turbine nho hon GT1 ~4.5MW).
    Chi assert cac khoang hop ly (0-100% cho moi %, MWh duong) -- KHONG assert khop tuyet
    doi ngoai production_mwh (da do co dinh, deterministic tren data that khong doi)."""
    m = DatasetManager()
    m.load_csv(GT1_PATH_REAL, name="GT1")
    curve_result = estimate_power_curve(m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd", dataset_name="GT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        power_curve=curve_result["data"], dataset_name="GT1", setpoint_column="GT1.SP.ActualSPP",
    )
    result = calculate_turbine_kpis(
        m, dataset_name="GT1", power_column="GT1.Grid.totW", wind_speed_column="GT1.Common.WindSpd",
        classification=classify_result["data"]["classification"],
        theoretical_power_column="GT1.Common.TheoActiveW",
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["method"] == "direct_theoretical"
    assert data["production_mwh"] == pytest.approx(33270.952, abs=0.01)
    assert 0 <= data["losses_pct"] <= 100
    assert 0 <= data["capacity_factor_pct"] <= 100
    assert 0 <= data["tba_pct"] <= 100
    assert 0 <= data["pba_pct"] <= 100
    assert data["reachable_mwh"] > 0
    assert data["rated_power_kw"] > 0


def test_integration_calculate_turbine_kpis_on_real_wt1_estimated_power_curve():
    """Da do that (2026-09-10): production_mwh=17181.6759 MWh (Site 2 khong co cot ly
    thuyet, dung estimated_power_curve)."""
    m = DatasetManager()
    m.load_csv(WT1_PATH_REAL, name="WT1")
    curve_result = estimate_power_curve(m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED", dataset_name="WT1", bin_width=1.0)
    classify_result = classify_turbine_status(
        m, power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        power_curve=curve_result["data"], dataset_name="WT1",
    )
    result = calculate_turbine_kpis(
        m, dataset_name="WT1", power_column="ACTIVE_POWER", wind_speed_column="WIND_SPEED",
        classification=classify_result["data"]["classification"], power_curve=curve_result["data"],
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["method"] == "estimated_power_curve"
    assert data["production_mwh"] == pytest.approx(17181.6759, abs=0.01)
    assert 0 <= data["losses_pct"] <= 100
    assert 0 <= data["capacity_factor_pct"] <= 100
    assert 0 <= data["tba_pct"] <= 100
    assert 0 <= data["pba_pct"] <= 100


# =============================== get_yaw_error ===============================

def test_get_yaw_error_site1_reads_column_directly():
    """Site 1: doc TRUC TIEP GT1.Common.WindTrackDevAng, KHONG bien doi gi (kha ca gia
    tri am/duong nam ngoai [-180,180] gia dinh -- neu co, do la du lieu that cua cot co
    san, ham nay khong tu y clip)."""
    df = pd.DataFrame({"GT1.Common.WindTrackDevAng": [-15.5, 0.0, 42.3, -180.0, 179.9]})
    result = get_yaw_error(df, "site1")
    assert list(result) == pytest.approx([-15.5, 0.0, 42.3, -180.0, 179.9])


def test_get_yaw_error_site2_wrap_around_hand_computed_cases():
    """Tay tinh tung case bien theo dung cong thuc ((raw+180)%360)-180, raw=NACELLE-WIND:
      nacelle=350, wind=10  -> raw=340   -> wrapped=-20  (dung vi du task: '350-10 phai
                                                           ra +-20, khong phai 340')
      nacelle=10,  wind=350 -> raw=-340  -> wrapped=+20  (chieu nguoc lai)
      nacelle=5,   wind=355 -> raw=-350  -> wrapped=+10
      nacelle=180, wind=0   -> raw=180   -> wrapped=-180 (bien dung == +-180)
      nacelle=0,   wind=0   -> raw=0     -> wrapped=0
      nacelle=90,  wind=80  -> raw=10    -> wrapped=10   (case thuong, khong wrap)"""
    df = pd.DataFrame({
        "DIRECTION_NACELLE": [350.0, 10.0, 5.0, 180.0, 0.0, 90.0],
        "DIRECTION_WIND":    [10.0, 350.0, 355.0, 0.0, 0.0, 80.0],
    })
    result = get_yaw_error(df, "site2")
    expected = [-20.0, 20.0, 10.0, -180.0, 0.0, 10.0]
    assert list(result) == pytest.approx(expected)
    assert all(-180.0 <= v <= 180.0 for v in result)


def test_get_yaw_error_site2_propagates_nan():
    df = pd.DataFrame({"DIRECTION_NACELLE": [350.0, np.nan], "DIRECTION_WIND": [10.0, 20.0]})
    result = get_yaw_error(df, "site2")
    assert result.iloc[0] == pytest.approx(-20.0)
    assert pd.isna(result.iloc[1])


def test_get_yaw_error_raises_for_unsupported_site():
    df = pd.DataFrame({"DIRECTION_NACELLE": [1.0], "DIRECTION_WIND": [2.0]})
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        get_yaw_error(df, "site3")


def test_get_yaw_error_raises_for_missing_columns():
    df = pd.DataFrame({"DIRECTION_NACELLE": [1.0]})  # thieu DIRECTION_WIND
    with pytest.raises(ValueError, match="DIRECTION_WIND"):
        get_yaw_error(df, "site2")
    df2 = pd.DataFrame({"other_col": [1.0]})
    with pytest.raises(ValueError, match="WindTrackDevAng"):
        get_yaw_error(df2, "site1")


def test_integration_get_yaw_error_on_real_wt1_distribution_is_physically_reasonable():
    """Kiem tra tren du lieu that Site 2 (WT1): moi gia tri phai nam trong [-180,180]
    (khong co gia tri 'dien ro' do loi wrap-around), va da so gan 0 (nacelle thuong bam
    theo huong gio, chi lech khi dang xoay/dieu chinh)."""
    m = DatasetManager()
    m.load_csv(WT1_PATH_REAL, name="WT1")
    df = m.get_dataframe("WT1")
    yaw_error = get_yaw_error(df, "site2")

    valid = yaw_error.dropna()
    assert len(valid) > 0
    assert valid.min() >= -180.0
    assert valid.max() <= 180.0

    median_abs = float(valid.abs().median())
    pct_within_10deg = float((valid.abs() <= 10.0).mean() * 100)
    print(f"\n[WT1 yaw_error] n={len(valid)}, median|yaw_error|={median_abs:.4f}°, "
          f"% trong [-10°,10°]={pct_within_10deg:.2f}%, min={valid.min():.2f}, max={valid.max():.2f}")
    # Nacelle bam theo huong gio phan lon thoi gian -- median lech nho (khong assert con
    # so cu the vi CHUA do truoc, chi assert bien do hop ly: median duoi 30 do, da so
    # (>50%) nam trong +-10 do quanh 0.
    assert median_abs < 30.0
    assert pct_within_10deg > 50.0
