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
