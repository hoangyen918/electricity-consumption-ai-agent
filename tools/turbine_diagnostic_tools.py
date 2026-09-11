"""
Tool diagnostic chuyên biệt turbine gió — verify lại đúng các vấn đề đã chốt trong
docs/turbine_data_audit.md. Tự nhận diện site (Site 1 GT1.* vs Site 2 ACTIVE_POWER/
WIND_SPEED) để áp đúng bộ rule/tên cột tương ứng, KHÔNG hardcode tên file.

Bài học từ audit: rule "power âm" KHÔNG dùng ngưỡng tuyệt đối "power < 0" — 100% giá
trị âm quan sát được ở cả 2 site có |giá trị| < 50kW (<1% rated power), là house/
parasitic load bình thường, không phải anomaly. Rule ở đây dùng ngưỡng magnitude-aware:
chỉ flag nếu power < -5% * rated_power, với rated_power tự ước lượng từ percentile 99.5
của các giá trị công suất dương quan sát được trong CHÍNH dataset đó (không hardcode số
tuyệt đối, không dùng số liệu nameplate ngoài).
"""
import numpy as np
import pandas as pd

from tools.dataset_manager import DatasetManager
from tools.data_cleaning_tools import detect_statistical_anomalies
from tools.tool_results import failure, success

CUT_IN_SPEED_MS = 3.0
CUT_OUT_SPEED_MS = 25.0
PITCH_MISMATCH_DEG = 5.0
ANEMOMETER_MISMATCH_PCT = 0.20
GEARBOX_OIL_ALARM_C = 90.0
STATOR_ALARM_C = 150.0
NEGATIVE_POWER_RATED_FRACTION = 0.05
ABSOLUTE_ZERO_SENTINEL_C = -273.0
MAX_EXAMPLES = 3

# Nhãn chính thức GT1.Common.OpSt — nguồn: data/wind_turbine/data1/note.txt (xác nhận
# 2026-09-08, xem docs/turbine_data_audit.md mục "Phụ lục — GT1.Common.OpSt"). Đặt tập
# trung ở đây để app.py (dashboard) và bất kỳ nơi nào khác dùng lại, không hardcode.
OPST_LABELS = {
    0: "Initialization", 1: "Stop", 2: "Standstill", 3: "Starting",
    4: "Run up", 5: "Power Production", 6: "Service",
}


def _dataset_not_found(manager: DatasetManager, dataset_name: str) -> str:
    return (
        f"Dataset '{dataset_name}' không tồn tại. Các dataset hiện có: "
        f"{manager.list_datasets()}"
    )


def _validate_column(manager: DatasetManager, column: str, dataset_name: str | None = None) -> str | None:
    columns = manager.get_columns(dataset_name)
    if column not in columns:
        return f"Column '{column}' not found in dataset. Valid columns are: {columns}"
    return None


def _detect_site(df: pd.DataFrame) -> str | None:
    if any(col.startswith("GT1.") for col in df.columns):
        return "site1"
    if "ACTIVE_POWER" in df.columns and "WIND_SPEED" in df.columns:
        return "site2"
    return None


def _power_wind_columns(site: str) -> tuple[str, str]:
    if site == "site1":
        return "GT1.Grid.totW", "GT1.Common.WindSpd"
    return "ACTIVE_POWER", "WIND_SPEED"


CATEGORICAL_MAX_UNIQUE = 31  # du cho hour(24)/month(12)/OpSt(7) nhung khong nham voi cot lien tuc


def _numeric_group_stats(series: pd.Series) -> dict:
    """describe() 1 nhom, dung chung cho explain_anomaly() va
    diagnose_theoretical_power_deviation() (tranh trung lap logic — xem note refactor
    o diagnose_theoretical_power_deviation)."""
    return {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in series.describe().items()}


def _categorical_group_stats(series: pd.Series) -> dict:
    """value_counts(normalize=True) 1 nhom (%), dung chung cho ca 2 ham tren."""
    counts = series.value_counts(normalize=True).mul(100).round(4)
    return {str(k): float(v) for k, v in counts.items()}


def _examples_from_mask(df: pd.DataFrame, mask: pd.Series, cols: list[str], n: int = MAX_EXAMPLES) -> list[dict]:
    subset = df.loc[mask, cols]
    examples = []
    for idx, row in subset.head(n).iterrows():
        example = {"row_index": int(idx)}
        for col in cols:
            val = row[col]
            example[col] = str(val) if isinstance(val, pd.Timestamp) else (float(val) if pd.notna(val) else None)
        examples.append(example)
    return examples


def _physical_violation_masks(df: pd.DataFrame, site: str, power_col: str, wind_col: str) -> dict:
    """
    Tính TOÀN BỘ mask thô (pd.Series[bool], full index, KHÔNG rút gọn) cho từng rule áp
    dụng được — dùng chung cho detect_physical_violations() (chỉ cần summary/examples)
    và classify_turbine_status() (cần mask thật để gộp thành nhãn "Measurement Error").
    Tách ra đây để KHÔNG viết lại rule logic ở 2 nơi (đúng yêu cầu Phase 3 chiều).

    Trả về dict[rule_name] = {"mask": pd.Series[bool], **metadata} — chỉ chứa rule THỰC
    SỰ áp dụng được cho site này (rule N/A không xuất hiện trong dict trả về, khác với
    detect_physical_violations() vẫn liệt kê rule N/A với applicable=False).
    """
    masks = {}

    positive_power = df.loc[df[power_col] > 0, power_col]
    if len(positive_power) > 0:
        rated_power = float(np.percentile(positive_power, 99.5))
        neg_threshold = -NEGATIVE_POWER_RATED_FRACTION * rated_power
        masks["negative_power_magnitude_aware"] = {
            "mask": df[power_col] < neg_threshold,
            "rated_power_estimate": round(rated_power, 4),
            "threshold_used": round(neg_threshold, 4),
        }

    masks["power_below_cutin"] = {
        "mask": (df[power_col] > 0) & (df[wind_col] < CUT_IN_SPEED_MS),
        "cutin_speed_ms": CUT_IN_SPEED_MS,
    }

    masks["power_above_cutout"] = {
        "mask": (df[power_col] > 0) & (df[wind_col] > CUT_OUT_SPEED_MS),
        "cutout_speed_ms": CUT_OUT_SPEED_MS,
    }

    pitch_cols = ["GT1.Pitch.Angle1", "GT1.Pitch.Angle2", "GT1.Pitch.Angle3"]
    if site == "site1" and all(c in df.columns for c in pitch_cols):
        pitch_max = df[pitch_cols].max(axis=1)
        pitch_min = df[pitch_cols].min(axis=1)
        masks["pitch_mismatch"] = {
            "mask": (pitch_max - pitch_min) > PITCH_MISMATCH_DEG,
            "threshold_deg": PITCH_MISMATCH_DEG,
        }

    if site == "site2" and "WIND_SPEED_2" in df.columns:
        denom = df["WIND_SPEED"].replace(0, np.nan)
        pct_diff = (df["WIND_SPEED"] - df["WIND_SPEED_2"]).abs() / denom
        masks["anemometer_mismatch"] = {
            "mask": pct_diff.fillna(0) > ANEMOMETER_MISMATCH_PCT,
            "threshold_pct": ANEMOMETER_MISMATCH_PCT,
        }

    temp_gearbox, temp_stator = "TEMPERATURE_GEARBOX_OIL", "TEMPERATURE_STATOR"
    if site == "site2" and temp_gearbox in df.columns and temp_stator in df.columns:
        masks["temperature_alarm"] = {
            "mask": (df[temp_gearbox] > GEARBOX_OIL_ALARM_C) | (df[temp_stator] > STATOR_ALARM_C),
            "gearbox_oil_threshold_c": GEARBOX_OIL_ALARM_C,
            "stator_threshold_c": STATOR_ALARM_C,
        }

    return masks


_PITCH_NA_REASON = "Chỉ áp dụng cho Site 1 (GT1), cần đủ 3 cột GT1.Pitch.Angle1/2/3."
_ANEMOMETER_NA_REASON = "Chỉ áp dụng cho Site 2, cần cột WIND_SPEED_2."
_TEMPERATURE_NA_REASON = "Chỉ áp dụng cho Site 2, cần cột TEMPERATURE_GEARBOX_OIL và TEMPERATURE_STATOR."


def detect_physical_violations(manager: DatasetManager, dataset_name: str | None = None) -> dict:
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        site = _detect_site(df)
        if site is None:
            return failure(
                "Không nhận diện được cấu trúc cột turbine gió (cần cột bắt đầu bằng "
                f"'GT1.' cho Site 1, hoặc cả 'ACTIVE_POWER' và 'WIND_SPEED' cho Site 2). "
                f"Cột hiện có: {list(df.columns)}"
            )

        power_col, wind_col = _power_wind_columns(site)
        n = len(df)
        raw_masks = _physical_violation_masks(df, site, power_col, wind_col)
        rules = {}

        def _build_rule_entry(name, example_cols, na_reason=None):
            if name not in raw_masks:
                rules[name] = {"applicable": False, "reason": na_reason}
                return
            info = raw_masks[name]
            mask = info["mask"]
            entry = {k: v for k, v in info.items() if k != "mask"}
            entry.update({
                "applicable": True,
                "count": int(mask.sum()),
                "pct": round(100 * mask.sum() / n, 4),
                "examples": _examples_from_mask(df, mask, example_cols),
            })
            rules[name] = entry

        if "negative_power_magnitude_aware" not in raw_masks:
            rules["negative_power_magnitude_aware"] = {
                "applicable": False,
                "reason": f"Không có giá trị {power_col} dương nào để ước lượng rated power.",
            }
        else:
            _build_rule_entry("negative_power_magnitude_aware", ["datetime", power_col, wind_col])

        _build_rule_entry("power_below_cutin", ["datetime", power_col, wind_col])
        _build_rule_entry("power_above_cutout", ["datetime", power_col, wind_col])
        _build_rule_entry("pitch_mismatch", ["datetime", "GT1.Pitch.Angle1", "GT1.Pitch.Angle2", "GT1.Pitch.Angle3"], _PITCH_NA_REASON)
        _build_rule_entry("anemometer_mismatch", ["datetime", "WIND_SPEED", "WIND_SPEED_2"], _ANEMOMETER_NA_REASON)
        _build_rule_entry("temperature_alarm", ["datetime", "TEMPERATURE_GEARBOX_OIL", "TEMPERATURE_STATOR"], _TEMPERATURE_NA_REASON)

        return success({
            "site_detected": site,
            "power_column": power_col,
            "wind_column": wind_col,
            "total_rows": n,
            "rules": rules,
        })
    except Exception as exc:
        return failure(str(exc))


def detect_sensor_issues(
    manager: DatasetManager,
    dataset_name: str | None = None,
    flatline_min_length: int = 6,
    gap_multiplier: float = 3.0,
) -> dict:
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        if "datetime" not in df.columns:
            return failure("Dataset không có cột 'datetime' (derive_time_columns không nhận diện được nguồn thời gian).")

        site = _detect_site(df)
        power_col = _power_wind_columns(site)[0] if site else None

        result = {"site_detected": site}

        # --- Flatline detection tren cot power (neu nhan dien duoc site) ---
        if power_col is not None:
            s = df[power_col].dropna()
            same_as_prev = s.eq(s.shift())
            group_id = (~same_as_prev).cumsum()
            run_lengths = s.groupby(group_id).size()
            qualifying = run_lengths[run_lengths >= flatline_min_length]
            total_rows_affected = int(qualifying.sum())
            examples = []
            for group_key in qualifying.index[:MAX_EXAMPLES]:
                idx_in_group = group_id[group_id == group_key].index
                examples.append({
                    "run_length": int(qualifying.loc[group_key]),
                    "value": float(s.loc[idx_in_group[0]]),
                    "start_row_index": int(idx_in_group[0]),
                })
            result["flatline"] = {
                "column_checked": power_col,
                "min_run_length": flatline_min_length,
                "n_runs": int(len(qualifying)),
                "total_rows_affected": total_rows_affected,
                "pct_rows": round(100 * total_rows_affected / len(df), 4),
                "examples": examples,
            }
        else:
            result["flatline"] = {"applicable": False, "reason": "Không nhận diện được site để chọn cột power."}

        # --- Timestamp gap: nguong = max(1 gio, gap_multiplier * timestep ky vong) ---
        diffs = df["datetime"].diff().dropna()
        mode_result = diffs.mode()
        expected_timestep = mode_result.iloc[0] if len(mode_result) > 0 else pd.Timedelta(0)
        gap_threshold = max(pd.Timedelta(hours=1), gap_multiplier * expected_timestep)
        gap_mask = diffs > gap_threshold
        gap_examples = []
        for idx in diffs.index[gap_mask][:MAX_EXAMPLES]:
            gap_examples.append({
                "gap_duration": str(diffs.loc[idx]),
                "ending_at": str(df["datetime"].loc[idx]),
            })
        result["timestamp_gap"] = {
            "expected_timestep": str(expected_timestep),
            "gap_threshold_used": str(gap_threshold),
            "count": int(gap_mask.sum()),
            "examples": gap_examples,
        }

        # --- Placeholder values: chi ap dung neu co cot TEMPERATURE* (Site 2) ---
        temp_cols = [c for c in df.columns if c.startswith("TEMPERATURE")]
        if temp_cols:
            abs_zero_mask = (df[temp_cols] <= ABSOLUTE_ZERO_SENTINEL_C).any(axis=1)
            simultaneous_zero_mask = (df[temp_cols] == 0.0).all(axis=1)
            result["placeholder_values"] = {
                "applicable": True,
                "temperature_columns_checked": temp_cols,
                "absolute_zero_sentinel": {
                    "threshold_c": ABSOLUTE_ZERO_SENTINEL_C,
                    "count": int(abs_zero_mask.sum()),
                    "pct": round(100 * abs_zero_mask.sum() / len(df), 4),
                    "examples": _examples_from_mask(df, abs_zero_mask, ["datetime"] + temp_cols),
                },
                "simultaneous_zero_all_temp_columns": {
                    "count": int(simultaneous_zero_mask.sum()),
                    "pct": round(100 * simultaneous_zero_mask.sum() / len(df), 4),
                    "examples": _examples_from_mask(df, simultaneous_zero_mask, ["datetime"] + temp_cols),
                },
            }
        else:
            result["placeholder_values"] = {"applicable": False, "reason": "Không có cột TEMPERATURE* trong dataset này."}

        return success(result)
    except Exception as exc:
        return failure(str(exc))


def diagnose_theoretical_power_deviation(
    manager: DatasetManager,
    dataset_name: str | None = None,
    threshold_pct: float = 10.0,
) -> dict:
    """
    Tool riêng cho vấn đề nghiêm trọng nhất còn mở trong audit: GT1 sản lượng thực
    (GT1.Grid.totW) vượt sản lượng lý thuyết (GT1.Common.TheoActiveW) hơn
    threshold_pct%. CHỈ trả dữ liệu mô tả so sánh nhóm vi phạm vs nhóm bình thường
    trên các field liên quan (OpSt, pitch, yaw, setpoint, thời gian) — KHÔNG tự kết
    luận nguyên nhân, để agent/user tự diễn giải giả thuyết.
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        required_cols = ["GT1.Grid.totW", "GT1.Common.TheoActiveW"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return failure(
                f"Tool này chỉ áp dụng cho dataset GT1 (cần cột {required_cols}). "
                f"Thiếu cột: {missing}. Cột hiện có: {list(df.columns)}"
            )

        mask = df["GT1.Grid.totW"] > (df["GT1.Common.TheoActiveW"] * (1 + threshold_pct / 100))
        violating = df[mask]
        normal = df[~mask]

        # Dung lai 2 leaf helper chung voi explain_anomaly() (_numeric_group_stats /
        # _categorical_group_stats) de tranh trung lap logic tinh describe()/
        # value_counts(). PHAN QUYET DINH cot nao la categorical/numeric van giu
        # HARDCODE rieng cho GT1 (KHONG dung chung heuristic nunique cua
        # explain_anomaly) — vi voi fixture test hien co, 1 so cot Pitch chi co 1
        # gia tri duy nhat trong data test nen se bi heuristic nunique hieu nham la
        # categorical, pha vo dung format output/test da co. Refactor sau xa hon (dung
        # chung ca phan quyet dinh) bi danh gia rui ro cao hon loi ich o buoc nay.
        def _numeric_compare(col):
            if col not in df.columns:
                return None
            return {"violating": _numeric_group_stats(violating[col]), "normal": _numeric_group_stats(normal[col])}

        def _categorical_compare(col):
            if col not in df.columns:
                return None
            return {"violating_pct": _categorical_group_stats(violating[col]), "normal_pct": _categorical_group_stats(normal[col])}

        comparison = {
            "GT1.Common.OpSt": _categorical_compare("GT1.Common.OpSt"),
            "GT1.Pitch.Angle1": _numeric_compare("GT1.Pitch.Angle1"),
            "GT1.Pitch.Angle2": _numeric_compare("GT1.Pitch.Angle2"),
            "GT1.Pitch.Angle3": _numeric_compare("GT1.Pitch.Angle3"),
            "GT1.Common.WindTrackDevAng": _numeric_compare("GT1.Common.WindTrackDevAng"),
            "GT1.SP.ActualSPP": _numeric_compare("GT1.SP.ActualSPP"),
            "hour": _categorical_compare("hour"),
            "month": _categorical_compare("month"),
        }
        comparison = {k: v for k, v in comparison.items() if v is not None}

        return success({
            "threshold_pct": threshold_pct,
            "violation_count": int(mask.sum()),
            "violation_pct": round(100 * mask.sum() / len(df), 4),
            "normal_count": int((~mask).sum()),
            "comparison": comparison,
        })
    except Exception as exc:
        return failure(str(exc))


def explain_anomaly(
    manager: DatasetManager,
    dataset_name: str | None = None,
    anomaly_mask_column: str | None = None,
    row_indices: list[int] | None = None,
    compare_columns: list[str] | None = None,
) -> dict:
    """
    Ham TONG QUAT — nhan mask/index bat ky, KHONG hardcode logic turbine gio. So sanh
    phan phoi cac cot compare_columns giua nhom "anomaly" (anomaly_mask_column=True
    hoac dung row_indices) va nhom con lai. Moi cot tu quyet dinh categorical (gia tri
    rac, vd OpSt/hour/month) hay numeric (gia tri lien tuc) theo so luong gia tri duy
    nhat (<= CATEGORICAL_MAX_UNIQUE la categorical) — KHAC voi
    diagnose_theoretical_power_deviation() (ham do hardcode danh sach cot theo domain
    GT1 co san, xem note refactor o do). Trả về mean/median/std moi nhom moi cot — KHONG
    co field 'conclusion', giu dung thiet ke da ap dung cho
    diagnose_theoretical_power_deviation (co test rieng xac nhan khong co field nay).
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        if (anomaly_mask_column is None) == (row_indices is None):
            return failure("Phải truyền ĐÚNG 1 trong 2: anomaly_mask_column hoặc row_indices (không được cả 2 hoặc không cái nào).")

        if anomaly_mask_column is not None:
            col_error = _validate_column(manager, anomaly_mask_column, dataset_name)
            if col_error:
                return failure(col_error)
            raw = df[anomaly_mask_column]
            non_bool_values = set(raw.dropna().unique()) - {True, False, 0, 1}
            if non_bool_values:
                return failure(
                    f"Cột '{anomaly_mask_column}' không phải dạng boolean/0-1 "
                    f"(có giá trị khác: {sorted(non_bool_values)[:10]})."
                )
            mask = raw.astype(bool)
        else:
            missing_idx = [i for i in row_indices if i not in df.index]
            if missing_idx:
                return failure(f"row_indices không tồn tại trong dataset: {missing_idx[:10]}")
            mask = df.index.isin(row_indices)
            mask = pd.Series(mask, index=df.index)

        if compare_columns is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            compare_columns = [c for c in numeric_cols if c != anomaly_mask_column]
        else:
            missing_cols = [c for c in compare_columns if c not in df.columns]
            if missing_cols:
                return failure(f"Cột không tồn tại: {missing_cols}. Cột hợp lệ: {list(df.columns)}")

        anomaly_group = df[mask]
        other_group = df[~mask]

        comparison = {}
        for col in compare_columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > CATEGORICAL_MAX_UNIQUE:
                comparison[col] = {
                    "type": "numeric",
                    "anomaly_group": _numeric_group_stats(anomaly_group[col]),
                    "other_group": _numeric_group_stats(other_group[col]),
                }
            else:
                comparison[col] = {
                    "type": "categorical",
                    "anomaly_group_pct": _categorical_group_stats(anomaly_group[col]),
                    "other_group_pct": _categorical_group_stats(other_group[col]),
                }

        return success({
            "anomaly_count": int(mask.sum()),
            "other_count": int((~mask).sum()),
            "anomaly_pct": round(100 * mask.sum() / len(df), 4),
            "compare_columns_used": compare_columns,
            "comparison": comparison,
        })
    except Exception as exc:
        return failure(str(exc))


MAX_POINT_CLASSIFICATION_EXAMPLES = 20


def estimate_power_curve(
    manager: DatasetManager,
    power_column: str,
    wind_speed_column: str,
    dataset_name: str | None = None,
    bin_width: float = 0.5,
    max_iterations: int = 5,
    convergence_threshold: float = 0.01,
    near_curve_iqr_multiplier: float = 2.0,
) -> dict:
    """
    Uoc luong power curve bang iterative median binning (dung phuong phap mo ta trong
    tai lieu PO024 da ingest: "iterative median estimation technique", ket qua duoc
    tuyen bo la ROBUST ke ca khi diem 'normal' chiem duoi 50% tong du lieu SCADA).
    KHONG kiem tra thay khuyen nghi bin_width cu the (IEC 61400-12-1 hay tuong tu)
    trong tai lieu KPI/PO024 da ingest khi tim lai — giu 0.5 m/s lam mac dinh hop ly.

    Thuat toan:
    1. Bin wind_speed_column theo bin_width (bin center = trung diem moi khoang).
    2. Vong lap dau: median moi bin tinh tren TOAN BO du lieu (active = tat ca).
    3. Voi median+IQR hien tai, phan loai MOI diem la "near curve" neu
       |power - median_bin| <= near_curve_iqr_multiplier * IQR_bin (IQR tinh tren tap
       active hien tai cua bin do). near_curve_iqr_multiplier va nguong "gan curve" KHONG
       co cong thuc chinh xac trong tai lieu da ingest — tu chon 2xIQR (quy uoc pho bien
       cho outlier robust), co the dieu chinh qua tham so.
    4. Tinh lai median CHI tren cac diem "near curve" (active moi = near_curve moi).
    5. Lap lai 3-4 toi khi hoi tu (median MOI bin thay doi tuong doi < convergence_threshold
       so voi vong truoc, chi so sanh cac bin CO du lieu o ca 2 vong) hoac het max_iterations.

    point_classification: KHONG tra ve 1 dict/dong goc (dataset thuc te co toi
    ~260.000 dong — se lam no ket qua tool qua kho de agent tieu thu, di nguoc pattern
    'tra ve tom tat + sample gioi han' da dung xuyen suot module nay). Thay vao do tra
    ve dem tong (near_curve_count/not_near_curve_count) + sample gioi han
    (MAX_POINT_CLASSIFICATION_EXAMPLES) cac diem KHONG near-curve de minh hoa. Neu can
    phan loai day du tung dong (vd cho classify_turbine_status sau nay), nen goi lai
    logic binning tren power_curve tra ve (nhe, chi vai chuc bin) thay vi doi ham nay
    tra ve toan bo array qua bien tool-result.

    Edge case: bin KHONG co diem nao (gap trong dai wind speed) -> median_power/
    std_power/iqr_power = None, n_points_used = 0, KHONG bi bo qua khoi power_curve
    (van xuat hien trong list, ghi ro n_points_total de phan biet "gap that" voi
    "bin co du lieu nhung bi loai het do khong near-curve").

    GIOI HAN DA PHAT HIEN KHI TEST (khong phai bug, la tinh chat toan hoc cua median):
    median CHI robust khi KHONG co 1 cum "nhieu" don le nao chiem >50% CUC BO trong 1
    bin cu the — khong chi can <50% TREN TOAN BO dataset. Da verify thuc nghiem: (1)
    nhieu "random scatter" that su (rai deu tren toan mien gia tri hop ly) -> phuc hoi
    tot o hau het cac bin, tru vung sat bien duoi (wind speed gan 0, noi tin hieu that
    cung gan 0 — sat mep mien nhieu nen bi lech nhe do bat doi xung). (2) nhieu tap
    trung dong nhat (vd toan bo "nhieu" đều co power thap bat ke wind speed, kieu
    stop/curtailment) -> that bai o HAU HET cac bin vi 1 cum nhieu do chinh no da
    chiem >50% cuc bo trong tung bin — median tu nhien chon nham cum nay. Y nghia
    thuc te: neu du lieu that co nhieu LOAI bat thuong khac nhau (stop/curtailment/
    under/over-production — dung nhu PO024 mo ta) thay vi 1 loai dong nhat, khong loai
    nao rieng le du chiem >50% mot bin -> phuong phap van robust nhu tuyen bo cua
    PO024. Neu chi co 1 loai nhieu dong nhat chiem da so, can kiem tra ky truoc khi tin
    ket qua o cac bin bi anh huong.
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        for col in (power_column, wind_speed_column):
            col_error = _validate_column(manager, col, dataset_name)
            if col_error:
                return failure(col_error)
            if not pd.api.types.is_numeric_dtype(df[col]):
                return failure(f"Column '{col}' không phải kiểu số.")

        working = df[[power_column, wind_speed_column]].dropna().copy()
        if len(working) == 0:
            return failure("Không có dòng nào có đủ dữ liệu (sau khi loại NaN ở power_column/wind_speed_column).")

        min_ws = float(working[wind_speed_column].min())
        max_ws = float(working[wind_speed_column].max())
        bin_start = np.floor(min_ws / bin_width) * bin_width
        bin_end = np.ceil(max_ws / bin_width) * bin_width + bin_width
        edges = np.arange(bin_start, bin_end + bin_width, bin_width)
        bin_centers = np.round((edges[:-1] + edges[1:]) / 2, 6)

        working["_bin"] = pd.cut(working[wind_speed_column], bins=edges, labels=bin_centers, include_lowest=True)
        total_counts = working.groupby("_bin", observed=True)[power_column].size()

        active_mask = pd.Series(True, index=working.index)
        prev_medians = None
        final_medians = final_iqrs = final_stds = final_counts = None
        converged = False
        iterations_run = 0

        for iteration in range(1, max_iterations + 1):
            iterations_run = iteration
            active_data = working[active_mask]
            grouped = active_data.groupby("_bin", observed=True)[power_column]
            medians = grouped.median()

            final_medians = medians
            final_iqrs = grouped.quantile(0.75) - grouped.quantile(0.25)
            final_stds = grouped.std()
            final_counts = grouped.size()

            if prev_medians is not None:
                common_bins = medians.index.intersection(prev_medians.index)
                if len(common_bins) > 0:
                    denom = prev_medians[common_bins].abs()
                    rel_change = ((medians[common_bins] - prev_medians[common_bins]).abs() / denom.replace(0, np.nan)).fillna(0)
                    if rel_change.max() < convergence_threshold:
                        converged = True
                        break

            prev_medians = medians

            if iteration == max_iterations:
                break

            bin_median_map = working["_bin"].map(medians).astype(float)
            bin_iqr_map = working["_bin"].map(final_iqrs).astype(float)
            has_median = bin_median_map.notna()
            deviation_all = (working[power_column] - bin_median_map).abs()
            near_curve = pd.Series(False, index=working.index)
            near_curve[has_median] = deviation_all[has_median] <= (near_curve_iqr_multiplier * bin_iqr_map[has_median].fillna(0))
            active_mask = near_curve

        power_curve = []
        for center in bin_centers:
            n_used = int(final_counts.get(center, 0)) if final_counts is not None else 0
            n_total = int(total_counts.get(center, 0))
            median_val = final_medians.get(center) if final_medians is not None else None
            power_curve.append({
                "wind_speed_bin_center": float(center),
                "median_power": (round(float(median_val), 4) if pd.notna(median_val) else None) if n_used > 0 else None,
                "std_power": (round(float(final_stds.get(center)), 4) if n_used > 1 and pd.notna(final_stds.get(center)) else None),
                "iqr_power": (round(float(final_iqrs.get(center)), 4) if n_used > 0 and pd.notna(final_iqrs.get(center)) else None),
                "n_points_used": n_used,
                "n_points_total": n_total,
            })

        # active_mask sau vong lap cuoi (truoc khi break) chinh la phan loai near-curve
        # da dung de tinh ra final_medians/final_counts o tren — luon nhat quan o ca 2
        # truong hop break (hoi tu som hoac het max_iterations).
        final_bin_median_map = working["_bin"].map(final_medians).astype(float) if final_medians is not None else pd.Series(np.nan, index=working.index)
        deviation_final = working[power_column] - final_bin_median_map
        not_near_curve_mask = ~active_mask
        not_near_curve_examples = []
        example_rows = working[not_near_curve_mask].head(MAX_POINT_CLASSIFICATION_EXAMPLES)
        for idx, row in example_rows.iterrows():
            dev = deviation_final.get(idx)
            not_near_curve_examples.append({
                "row_index": int(idx),
                wind_speed_column: float(row[wind_speed_column]),
                power_column: float(row[power_column]),
                "bin_median_power": (round(float(final_bin_median_map.get(idx)), 4) if pd.notna(final_bin_median_map.get(idx)) else None),
                "deviation_from_curve": (round(float(dev), 4) if pd.notna(dev) else None),
            })

        return success({
            "power_column": power_column,
            "wind_speed_column": wind_speed_column,
            "bin_width": bin_width,
            "near_curve_iqr_multiplier": near_curve_iqr_multiplier,
            "power_curve": power_curve,
            "convergence_info": {
                "iterations_run": iterations_run,
                "converged": converged,
                "max_iterations": max_iterations,
                "convergence_threshold": convergence_threshold,
            },
            "point_classification_summary": {
                "total_points": len(working),
                "near_curve_count": int(active_mask.sum()),
                "not_near_curve_count": int(not_near_curve_mask.sum()),
                "near_curve_pct": round(100 * active_mask.sum() / len(working), 4),
                "not_near_curve_examples_sample": not_near_curve_examples,
                "note": "Khong tra ve day du tung dong (dataset co the co hang tram nghin dong) — xem docstring.",
            },
        })
    except Exception as exc:
        return failure(str(exc))


# 8 nhan Meteodyn (PO024) — thu tu chinh la THU TU UU TIEN cua classify_turbine_status():
# diem nao khop rule truoc thi dung, KHONG check tiep cac rule sau.
CLASSIFICATION_LABELS = [
    "Measurement Error", "Stop", "Curtailment", "Under Production",
    "Over Production", "Partial Stop", "Partial Curtailment", "Normal",
]


def classify_turbine_status(
    manager: DatasetManager,
    power_column: str,
    wind_speed_column: str,
    power_curve: dict,
    dataset_name: str | None = None,
    setpoint_column: str | None = None,
    stop_power_threshold_pct: float = 0.01,
    partial_stop_power_threshold_pct: float = 0.05,
    curtailment_tracking_tolerance_pct: float = 0.05,
    partial_curtailment_tracking_tolerance_pct: float = 0.15,
    curtailment_setpoint_below_curve_pct: float = 0.9,
    deviation_std_threshold: float = 2.0,
) -> dict:
    """
    Phân loại 8 nhãn Meteodyn cho từng dòng, dựa trên độ lệch so với power curve đã
    ước lượng SẴN (power_curve = field "data" từ kết quả estimate_power_curve(), TÁI
    DÙNG — hàm này KHÔNG tự gọi lại estimate_power_curve()).

    THỨ TỰ ƯU TIÊN (điểm khớp rule nào trước thì dừng, không check rule sau) —
    xem CLASSIFICATION_LABELS:
    1. Measurement Error: tái dùng TOÀN BỘ mask thô từ _physical_violation_masks()
       (cùng hàm nội bộ detect_physical_violations() dùng — KHÔNG viết lại rule logic).
       Nếu dataset không nhận diện được site (GT1.*/ACTIVE_POWER+WIND_SPEED), bỏ qua
       bước này (không có rule nào để tái dùng), không raise lỗi.
    2. Stop: power < stop_power_threshold_pct × rated_power VÀ wind_speed > cut-in
       (CUT_IN_SPEED_MS, cùng hằng số detect_physical_violations() dùng — không hardcode
       2 nơi). rated_power = max(median_power) trong power_curve (bin cao nhất/gần rated).
       ĐÃ VALIDATE (2026-09-08, xem docs/turbine_data_audit.md): nhãn "Stop" đo khái niệm
       RỘNG "công suất gần 0 do wind>cut-in" — đối chiếu với GT1.Common.OpSt cho thấy đây
       gộp cả OpSt=Service/Starting/Standstill/Stop (4 trạng thái vận hành khác nhau cùng
       biểu hiện công suất gần 0, chỉ 7.88% khớp đúng OpSt∈{Stop,Standstill}). Rule hoạt
       động ĐÚNG thiết kế — 100% điểm bị "lệch" (OpSt=Service/Starting) vẫn nằm trong đúng
       ngưỡng power khi kiểm tra lại (|totW| ≤ 1%×rated ở mọi điểm) — đây là khác biệt độ
       chi tiết taxonomy giữa rule và mã OpSt, KHÔNG phải bug.
    3. Curtailment: CHỈ áp dụng nếu setpoint_column có giá trị (Site 1) — Site 2 KHÔNG
       BAO GIỜ được gán nhãn này (thiếu tín hiệu setpoint đáng tin cậy để phân biệt với
       Under Production thật). Điều kiện: |power-setpoint|/setpoint <
       curtailment_tracking_tolerance_pct VÀ setpoint < median_power_bin ×
       curtailment_setpoint_below_curve_pct.
    4. Under Production: (power - median_power_bin) / std_power_bin <
       -deviation_std_threshold. Bin không có std hợp lệ (std=None/0, vd bin rỗng hoặc
       chỉ 1 điểm) → bỏ qua rule này cho dòng đó (không raise).
    5. Over Production: ngược lại, > +deviation_std_threshold.
    6. Partial Stop / Partial Curtailment: biến thể "nhẹ hơn" — ngưỡng LỎNG HƠN
       (partial_stop_power_threshold_pct=0.05 thay vì 0.01; partial_curtailment_tracking_
       tolerance_pct=0.15 thay vì 0.05). QUAN TRỌNG: đây là ngưỡng HEURISTIC tự chọn,
       KHÔNG có nguồn chính thức xác nhận ranh giới chính xác Full vs Partial — cần domain
       expert review sau. Do được check SAU Under/Over Production (rule 4-5) trong thứ tự
       ưu tiên, phần lớn điểm "công suất rất thấp" đã bị rule 4 bắt trước (deviation âm sâu
       so với bin) — Partial Stop/Partial Curtailment trong thực tế có thể hiếm khi kích
       hoạt; đây là hệ quả TRỰC TIẾP của thứ tự ưu tiên đã chỉ định, không phải bug.
       ĐÃ VALIDATE (2026-09-08, xem docs/turbine_data_audit.md): ngưỡng Partial Stop hiện
       tại (power < 5%×rated, TUYỆT ĐỐI — KHÔNG dùng cơ chế deviation-from-curve như
       Under/Over Production) cho agreement RẤT THẤP với OpSt — chỉ 0.25% điểm khớp
       OpSt∈{Stop,Standstill}, trong khi 92.98% lại rơi vào OpSt=Power Production. Đã thử
       áp persistence rule (group_anomaly_windows(), min_consecutive_points=3): loại được
       39.63% điểm (nhiễu ngắn hạn <3 điểm liên tiếp), nhưng ở các window CÒN LẠI agreement
       KHÔNG cải thiện — thậm chí tệ hơn (94.60% Power Production, chỉ 0.06% Stop/
       Standstill). Kết luận: rule đang bắt các đợt sụt công suất tạm thời trong vận hành
       BÌNH THƯỜNG (gió yếu dần), không phải hiện tượng bán-dừng thật — persistence rule
       không sửa được vấn đề này. CẦN domain expert review định nghĩa ngưỡng trước khi
       dùng nhãn này cho quyết định vận hành thật.
    7. Normal: còn lại. Dòng rơi vào bin KHÔNG có dữ liệu trong power_curve (median=None,
       vd wind speed hiếm/ngoài phạm vi ước lượng ban đầu) mặc định về Normal nếu không
       khớp rule nào khác — giới hạn đã biết, không có nhãn "Unknown" riêng theo đúng 8
       nhãn Meteodyn yêu cầu.

    Trả về: {"classification": {row_index: label}, "summary": {label: {count, pct}}}.
    LƯU Ý: classification là dict ĐẦY ĐỦ TỪNG DÒNG (không rút gọn/sample) vì
    group_anomaly_windows() cần dữ liệu đầy đủ để gộp cửa sổ thời gian — khác với các
    tool khác trong module này (vd estimate_power_curve) chỉ trả summary+sample. Với
    dataset lớn (~260.000 dòng), dict này có thể vài MB trong bộ nhớ Python — chấp nhận
    được vì được tiêu thụ trực tiếp trong process (group_anomaly_windows), không phải để
    hiển thị nguyên văn cho agent/LLM.
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        for col in (power_column, wind_speed_column):
            col_error = _validate_column(manager, col, dataset_name)
            if col_error:
                return failure(col_error)
        if setpoint_column is not None:
            col_error = _validate_column(manager, setpoint_column, dataset_name)
            if col_error:
                return failure(col_error)

        power_curve_list = power_curve.get("power_curve")
        bin_width = power_curve.get("bin_width")
        if not power_curve_list or not bin_width:
            return failure(
                "power_curve không hợp lệ — cần truyền field 'data' từ kết quả "
                "estimate_power_curve() (có 'power_curve' và 'bin_width'), không phải "
                "toàn bộ dict {'ok':..,'data':..}."
            )

        median_lookup = pd.Series({b["wind_speed_bin_center"]: b["median_power"] for b in power_curve_list})
        std_lookup = pd.Series({b["wind_speed_bin_center"]: b["std_power"] for b in power_curve_list})
        positive_medians = median_lookup.dropna()
        positive_medians = positive_medians[positive_medians > 0]
        if len(positive_medians) == 0:
            return failure("power_curve không có bin nào có median_power hợp lệ để ước lượng rated power.")
        rated_power = float(positive_medians.max())

        n = len(df)
        power = df[power_column]
        wind = df[wind_speed_column]
        bin_centers = np.round(np.floor(wind / bin_width) * bin_width + bin_width / 2, 6)
        median_series = pd.Series(median_lookup.reindex(bin_centers.values).values, index=df.index)
        std_series = pd.Series(std_lookup.reindex(bin_centers.values).values, index=df.index)

        site = _detect_site(df)
        measurement_error_mask = pd.Series(False, index=df.index)
        if site is not None:
            raw_masks = _physical_violation_masks(df, site, power_column, wind_speed_column)
            for info in raw_masks.values():
                measurement_error_mask = measurement_error_mask | info["mask"]

        labels = pd.Series("Normal", index=df.index, dtype=object)
        assigned = pd.Series(False, index=df.index)

        m = measurement_error_mask & ~assigned
        labels[m] = "Measurement Error"
        assigned = assigned | m

        m = (~assigned) & (power < stop_power_threshold_pct * rated_power) & (wind > CUT_IN_SPEED_MS)
        labels[m] = "Stop"
        assigned = assigned | m

        if setpoint_column is not None:
            setpoint = df[setpoint_column]
            valid_setpoint = setpoint.notna() & (setpoint != 0) & median_series.notna()
            tracks_setpoint_tight = ((power - setpoint).abs() / setpoint.abs()) < curtailment_tracking_tolerance_pct
            setpoint_below_curve = setpoint < (median_series * curtailment_setpoint_below_curve_pct)

            m = (~assigned) & valid_setpoint & tracks_setpoint_tight.fillna(False) & setpoint_below_curve.fillna(False)
            labels[m] = "Curtailment"
            assigned = assigned | m

        valid_std = std_series.notna() & (std_series > 0) & median_series.notna()
        deviation_z = (power - median_series) / std_series

        m = (~assigned) & valid_std & (deviation_z < -deviation_std_threshold)
        labels[m] = "Under Production"
        assigned = assigned | m

        m = (~assigned) & valid_std & (deviation_z > deviation_std_threshold)
        labels[m] = "Over Production"
        assigned = assigned | m

        m = (~assigned) & (power < partial_stop_power_threshold_pct * rated_power) & (wind > CUT_IN_SPEED_MS)
        labels[m] = "Partial Stop"
        assigned = assigned | m

        if setpoint_column is not None:
            tracks_setpoint_loose = ((power - setpoint).abs() / setpoint.abs()) < partial_curtailment_tracking_tolerance_pct
            m = (~assigned) & valid_setpoint & tracks_setpoint_loose.fillna(False) & setpoint_below_curve.fillna(False)
            labels[m] = "Partial Curtailment"
            assigned = assigned | m

        # con lai giu nguyen "Normal" (khoi tao ban dau)

        counts = labels.value_counts()
        summary = {lbl: {"count": int(cnt), "pct": round(100 * cnt / n, 4)} for lbl, cnt in counts.items()}

        return success({
            "site_detected": site,
            "rated_power_estimate": round(rated_power, 4),
            "curtailment_applicable": setpoint_column is not None,
            "classification": {int(idx): lbl for idx, lbl in labels.items()},
            "summary": summary,
        })
    except Exception as exc:
        return failure(str(exc))


def group_anomaly_windows(
    manager: DatasetManager,
    classification: dict,
    timestamp_column: str,
    dataset_name: str | None = None,
    min_consecutive_points: int = 3,
    exclude_labels: list[str] | None = None,
) -> dict:
    """
    Gộp các điểm LIÊN TIẾP THEO THỜI GIAN cùng nhãn (từ classify_turbine_status(),
    không tính lại) thành 1 khoảng thời gian. Tự sắp xếp lại theo timestamp_column
    (không giả định dataset đã sort sẵn). Chỉ giữ khoảng có >= min_consecutive_points.

    exclude_labels mặc định ["Normal"] nếu để None (KHÔNG dùng default argument dạng
    list trực tiếp trong signature — tránh lỗi mutable default argument kinh điển của
    Python).
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))
        col_error = _validate_column(manager, timestamp_column, dataset_name)
        if col_error:
            return failure(col_error)

        exclude_labels = exclude_labels if exclude_labels is not None else ["Normal"]

        labels_series = pd.Series(classification)
        labels_series.index = labels_series.index.astype(df.index.dtype) if len(labels_series) else labels_series.index
        ordered_index = df[timestamp_column].sort_values().index
        ordered_index = [i for i in ordered_index if i in labels_series.index]
        if not ordered_index:
            return failure("Không có row_index nào trong classification khớp với dataset (kiểm tra lại classification có đúng lấy từ classify_turbine_status() trên CÙNG dataset_name này không).")

        windows = []
        current_label = None
        run_indices = []

        def _flush():
            if current_label is not None and current_label not in exclude_labels and len(run_indices) >= min_consecutive_points:
                start_time = df[timestamp_column].loc[run_indices[0]]
                end_time = df[timestamp_column].loc[run_indices[-1]]
                windows.append({
                    "label": current_label,
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration": str(end_time - start_time),
                    "n_points": len(run_indices),
                })

        for idx in ordered_index:
            lbl = labels_series.loc[idx]
            if lbl == current_label:
                run_indices.append(idx)
            else:
                _flush()
                current_label = lbl
                run_indices = [idx]
        _flush()

        return success({
            "windows": windows,
            "total_windows": len(windows),
            "min_consecutive_points": min_consecutive_points,
            "exclude_labels": exclude_labels,
        })
    except Exception as exc:
        return failure(str(exc))


GATE_4A_MIN_OVERLAP_PCT = 80.0
GATE_4A_MEANINGFUL_SAMPLE_SIZE = 10
GATE_4A_MIN_MEANINGFUL_COUNT = 3


def detect_anomalies_ml(
    manager: DatasetManager,
    power_column: str,
    wind_speed_column: str,
    classification: dict,
    dataset_name: str | None = None,
    pitch_column: str | None = None,
    wind_speed_std_column: str | None = None,
    train_fraction: float = 0.8,
    z_score_threshold: float = 3.0,
    random_state: int = 42,
) -> dict:
    """
    Isolation Forest phát hiện bất thường, CHỈ dùng nếu vượt Gate 4a — so với
    detect_statistical_anomalies() (z-score, Phase 2) trên CÙNG dataset. Nếu KHÔNG
    vượt gate, tự động fallback về detect_statistical_anomalies() nguyên bản (rule-based)
    — người gọi tool KHÔNG cần biết đằng sau dùng ML hay rule-based.

    ĐÃ VALIDATE THẬT (2026-09-08) trên cả 2 site GT1/WT1 — KẾT QUẢ: **Gate 4a FAIL cả
    2 site**, mặc dù kiểm tra ý nghĩa vật lý riêng lẻ PASS (7/10 và 5/10 điểm IF-flag-
    riêng có classification != Normal). Chi tiết:
    - GT1: overlap=0.00% (z-score threshold=3.0 trên GT1.Grid.totW flag ĐÚNG 0 điểm
      trong toàn bộ 233.953 dòng — không phải bug, do std cột này (~1530) quá lớn so
      với range thực tế (-44.5 → 4525.2), nên |z|>3 không bao giờ xảy ra — baseline
      z-score bị suy biến cho chính cột/ngưỡng này, không phải IF kém).
    - WT1: overlap=5.38% (5.150/95.788 điểm IF-flag trùng z-score) — xa dưới 80%.
    Hàm này VẪN implement ĐẦY ĐỦ logic Isolation Forest + gate-check THẬT (không
    hardcode nhánh fallback) để tổng quát cho dataset khác có thể vượt gate trong
    tương lai — nhưng trên GT1/WT1 hiện tại, kết quả THỰC TẾ khi gọi hàm này sẽ LUÔN
    là fallback, vì gate thất bại thật, không phải giả lập.

    THỬ LẠI LẦN 2 (2026-09-11, script khảo sát KHÔNG commit vào repo, KHÔNG tích hợp
    vào hàm này — chỉ ghi nhận kết quả): train Isolation Forest trên deviation_from_curve
    (residual = power - median_power_tại_bin, TÁI DÙNG đúng công thức classify_turbine_
    status() dùng) thay vì (power, wind_speed) thô, thử cả 2 biến thể 1D (chỉ deviation)
    và 2D (deviation + wind_speed), so với 2 baseline z-score ĐỘC LẬP (z-score trên power
    thô — như lần 1; VÀ z-score MỚI trên chính deviation_from_curve). KẾT QUẢ: **Gate 4a
    VẪN FAIL cả 4 tổ hợp** (2 site × 2 biến thể) — overlap 0.00%-5.97%, xa dưới 80%:
    - GT1: baseline z-score-trên-deviation KHÔNG còn suy biến như baseline z-score-trên-
      power (flag 1.43% điểm thay vì 0%) — nhưng overlap với IF vẫn chỉ 4.35%-5.97%.
    - WT1: overlap 2.44%-5.24% (baseline power) và 2.61%-3.84% (baseline deviation).
    NGUYÊN NHÂN CẤU TRÚC (không chỉ do chọn sai feature): contamination ước lượng từ %
    không-Normal trong classification cao bất thường (22.85% GT1, 28.10% WT1) vì gồm cả
    trạng thái vận hành hợp lệ (Curtailment/Stop/Partial Stop...), không chỉ điểm bất
    thường thật — buộc Isolation Forest flag 24%-37% tổng số điểm, trong khi z-score chỉ
    flag <2% (đuôi cực trị thống kê). 2 phương pháp đo 2 khái niệm khác nhau về bản chất
    ("~1/4-1/3 dữ liệu lệch nhẹ" vs "điểm cực trị hiếm") nên overlap thấp gần như là hệ
    quả tất yếu của cách ước lượng contamination hiện tại, không chỉ do không gian
    feature. Đây là BẰNG CHỨNG THỨ 2 (độc lập với lần 1, feature khác hẳn) củng cố quyết
    định KHÔNG dùng ML cho phát hiện bất thường trên 2 dataset này — nếu muốn thử lại lần
    3, nên xem xét đổi cách ước lượng contamination (vd percentile cố định thay vì %
    không-Normal) trước khi đổi feature lần nữa.

    classification: dict {row_index: label} từ classify_turbine_status() (TÁI DÙNG,
    không tính lại) — dùng để (a) lọc tập "Normal" làm train set, (b) ước lượng
    contamination = % không-Normal trong CHÍNH dataset (không hardcode 0.1 mặc định
    sklearn), (c) kiểm tra ý nghĩa vật lý của điểm IF-flag-riêng (Gate 4a phần 2).

    sklearn được import LAZY bên trong hàm (không import ở đầu module) — tránh rủi ro
    đã gặp thật khi phát triển: scikit-learn transitively cần scipy.sparse.csgraph,
    1 số bản build scipy bị Windows Application Control policy chặn DLL (đã pin
    scipy==1.11.4 trong requirements.txt để tránh, nhưng vẫn phòng hờ import lazy để
    lỗi sklearn không làm sập TOÀN BỘ module turbine_diagnostic_tools.py).
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        for col in (power_column, wind_speed_column):
            col_error = _validate_column(manager, col, dataset_name)
            if col_error:
                return failure(col_error)

        def _fallback(reason: str) -> dict:
            fallback_result = detect_statistical_anomalies(
                manager, column=power_column, dataset_name=dataset_name,
                method="zscore", threshold=z_score_threshold,
            )
            if not fallback_result["ok"]:
                return fallback_result
            fallback_result["data"]["method_used"] = "fallback_statistical_zscore"
            fallback_result["data"]["ml_gate_reason"] = reason
            return fallback_result

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            return _fallback(f"Không import được scikit-learn ({exc}) — dùng fallback rule-based.")

        feature_cols = [power_column, wind_speed_column]
        if pitch_column is not None:
            col_error = _validate_column(manager, pitch_column, dataset_name)
            if col_error:
                return failure(col_error)
            feature_cols.append(pitch_column)
        if wind_speed_std_column is not None:
            col_error = _validate_column(manager, wind_speed_std_column, dataset_name)
            if col_error:
                return failure(col_error)
            feature_cols.append(wind_speed_std_column)

        n = len(df)
        split_idx = int(n * train_fraction)
        class_series = pd.Series(classification)
        class_series.index = class_series.index.astype(df.index.dtype) if len(class_series) else class_series.index
        class_series = class_series.reindex(df.index)

        pct_not_normal = (class_series != "Normal").mean()
        contamination = min(max(round(float(pct_not_normal), 4), 0.001), 0.5)

        features_all = df[feature_cols]
        valid_mask = features_all.notna().all(axis=1)
        train_mask = (pd.Series(df.index, index=df.index) < split_idx) & (class_series == "Normal") & valid_mask

        if train_mask.sum() < 30:
            return _fallback(f"Không đủ điểm 'Normal' trong tập train để fit Isolation Forest đáng tin cậy (chỉ {int(train_mask.sum())} điểm, cần >=30).")

        iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        iso.fit(features_all[train_mask])

        preds = pd.Series(1, index=df.index)
        preds[valid_mask] = iso.predict(features_all[valid_mask])
        is_anomaly_if = preds == -1

        # z-score tren power_column, cong thuc y het detect_statistical_anomalies() (method="zscore")
        power_series = df[power_column].dropna()
        mean_p, std_p = power_series.mean(), power_series.std()
        if std_p == 0 or pd.isna(std_p):
            return _fallback(f"Column '{power_column}' có std=0, không thể tính z-score để gate-check — dùng fallback.")
        z = (df[power_column] - mean_p) / std_p
        is_anomaly_zscore = (z.abs() > z_score_threshold).fillna(False)

        n_if = int(is_anomaly_if.sum())
        if n_if == 0:
            return _fallback("Isolation Forest không flag điểm nào — không có gì để so gate, dùng fallback.")

        overlap_pct = float(100 * (is_anomaly_if & is_anomaly_zscore).sum() / n_if)
        gate_overlap_pass = bool(overlap_pct >= GATE_4A_MIN_OVERLAP_PCT)

        if_only_idx = df.index[is_anomaly_if & (~is_anomaly_zscore)].tolist()
        rng = np.random.RandomState(random_state)
        sample_size = min(GATE_4A_MEANINGFUL_SAMPLE_SIZE, len(if_only_idx))
        sample_idx = rng.choice(if_only_idx, size=sample_size, replace=False) if sample_size > 0 else np.array([])
        meaningful_count = int(sum(class_series.loc[i] != "Normal" for i in sample_idx))
        gate_meaning_pass = meaningful_count >= GATE_4A_MIN_MEANINGFUL_COUNT

        gate_info = {
            "overlap_pct_with_zscore": round(overlap_pct, 4),
            "overlap_gate_threshold_pct": GATE_4A_MIN_OVERLAP_PCT,
            "overlap_gate_pass": gate_overlap_pass,
            "meaningful_sample_size": int(sample_size),
            "meaningful_count": meaningful_count,
            "meaningful_gate_min": GATE_4A_MIN_MEANINGFUL_COUNT,
            "meaningful_gate_pass": gate_meaning_pass,
        }

        if not (gate_overlap_pass and gate_meaning_pass):
            result = _fallback(
                f"Gate 4a FAIL: overlap={overlap_pct:.2f}% (cần >={GATE_4A_MIN_OVERLAP_PCT}%), "
                f"meaningful={meaningful_count}/{sample_size} (cần >={GATE_4A_MIN_MEANINGFUL_COUNT})."
            )
            if result["ok"]:
                result["data"]["gate_4a"] = gate_info
            return result

        examples = _examples_from_mask(df, is_anomaly_if, ["datetime", power_column, wind_speed_column])
        return success({
            "method_used": "isolation_forest",
            "gate_4a": gate_info,
            "contamination_used": contamination,
            "feature_columns": feature_cols,
            "anomaly_count": n_if,
            "anomaly_pct": round(100 * n_if / n, 4),
            "examples": examples,
        })
    except Exception as exc:
        return failure(str(exc))


def _power_curve_interpolate(power_curve: dict, wind_speed_values) -> np.ndarray:
    """Baseline 'ngay tho': noi suy tuyen tinh giua cac median_power cua power_curve
    (tu estimate_power_curve(), TAI DUNG). Ngoai mien bin da biet -> np.interp tu dong
    clamp ve gia tri bien gan nhat (hanh vi mac dinh cua numpy, khong tu ngoai suy)."""
    valid_bins = sorted(
        (b["wind_speed_bin_center"], b["median_power"])
        for b in power_curve.get("power_curve", []) if b.get("median_power") is not None
    )
    if not valid_bins:
        raise ValueError("power_curve không có bin nào có median_power hợp lệ để nội suy baseline.")
    xs = [b[0] for b in valid_bins]
    ys = [b[1] for b in valid_bins]
    return np.interp(wind_speed_values, xs, ys)


# Nguong loai diem "gan 0" khi tinh MAPE cho predict_power() — KHOP voi nguong da dung
# o "Vấn đề 10" (docs/turbine_data_audit.md): |cong suat am/gan 0| <50kW la house/
# parasitic load binh thuong (<1% rated ca 2 site GT1 va WT1-4), KHONG phai muc cong
# suat phat dien binh thuong. Neu khong loc, cac diem nay (actual gan 0, vd 1kW) lam
# MAPE bi thoi phong phi ly (actual=1kW, predict=5kW -> loi diem do = 400%), khong phan
# anh dung chat luong model o vung cong suat phat dien binh thuong (hang tram-nghin kW).
MAPE_MIN_ACTUAL_POWER_KW = 50.0


def _mape(actual: np.ndarray, pred: np.ndarray, min_actual_threshold: float) -> float | None:
    """MAPE (%) CHI tinh tren cac diem |actual| > min_actual_threshold (xem
    MAPE_MIN_ACTUAL_POWER_KW). Tra ve None neu khong con diem nao sau khi loc (vd toan
    bo tap test deu la diem gan 0 — hiem nhung co the xay ra voi dataset nho/loc manh)."""
    mask = np.abs(actual) > min_actual_threshold
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def predict_power(
    manager: DatasetManager,
    power_column: str,
    wind_speed_column: str,
    power_curve: dict,
    classification: dict,
    dataset_name: str | None = None,
    feature_columns: list[str] | None = None,
    train_fraction: float = 0.8,
    random_state: int = 42,
    include_predictions: bool = False,
) -> dict:
    """
    XGBoost dự đoán power_column từ wind_speed_column + feature_columns bổ sung (pitch/
    yaw/temperature — KHÔNG bao giờ dùng TheoActiveW làm feature, tránh leakage vì đó
    gần như là đáp án). Baseline so sánh: nội suy tuyến tính power curve
    (estimate_power_curve(), TÁI DÙNG). CHỈ dùng XGBoost nếu vượt CẢ Gate 4b (RMSE thấp
    hơn baseline) VÀ Gate 4c (wind_speed xếp hạng #1 hoặc #2 feature importance) — không
    đạt 1 trong 2 → tự động fallback về baseline nội suy, người gọi KHÔNG cần biết đằng
    sau dùng gì.

    ĐÃ VALIDATE THẬT (2026-09-08) trên cả GT1 và WT1 — CẢ 2 GATE ĐỀU PASS CẢ 2 SITE:
    - GT1 (feature: WindSpd, Pitch.Angle1, WindTrackDevAng): XGBoost RMSE=269.11 <
      baseline RMSE=305.15 (R²=0.9636 vs 0.9532). wind_speed xếp #1 (importance=0.9967).
    - WT1 (feature: WIND_SPEED, PITCH, WIND_SPEED_STD, TEMPERATURE_STATOR — TEMPERATURE_
      STATOR được chọn vì correlation_analysis (Phase 3 sáng) cho r=0.43, cao nhất trong
      các cột nhiệt độ): XGBoost RMSE=135.12 < baseline RMSE=151.58 (R²=0.9070 vs
      0.8830). wind_speed xếp #1 (importance=0.9894).
    Khác với detect_anomalies_ml() (Gate 4a FAIL cả 2 site), predict_power() dùng
    XGBoost THẬT trên cả 2 site khi gọi với đúng feature_columns nêu trên — đây không
    phải mô phỏng, là kết quả đo thật.

    Huấn luyện CHỈ trên điểm "Normal" (classify_turbine_status(), TÁI DÙNG) trong
    train_fraction đầu (theo thời gian) — nhất quán phương pháp với estimate_power_curve()
    cũng ước lượng từ điểm "gần curve". Dự đoán trên TOÀN BỘ tập test (không chỉ Normal)
    để so sánh công bằng với thực tế vận hành.

    Gate 4c nghiêm ngặt: nếu wind_speed xếp hạng thấp bất thường (>2), DỪNG dùng XGBoost
    NGAY CẢ KHI Gate 4b pass — dấu hiệu nghi ngờ bug preprocessing (lệch thời gian giữa
    feature/target, feature bị scale sai...), không chỉ đơn thuần "model kém".

    metrics/baseline_metrics có thêm "mape" (%, cả XGBoost lẫn baseline) — CHỈ để hiển
    thị thêm thông tin, KHÔNG dùng để quyết định gate_4b_pass (vẫn chốt theo RMSE như
    trước). MAPE chỉ tính trên điểm |actual| > MAPE_MIN_ACTUAL_POWER_KW (50kW) — bỏ qua
    điểm gần 0 (turbine Stop/gió yếu), xem docstring MAPE_MIN_ACTUAL_POWER_KW/_mape() ở
    trên; nếu không lọc, các điểm actual gần 0 sẽ thổi phồng MAPE phi lý (vd actual=1kW,
    predict=5kW → lỗi điểm đó = 400%, không phản ánh đúng chất lượng model ở vùng công
    suất phát điện bình thường). Có thể trả về None nếu không còn điểm nào sau khi lọc.

    MAPE THẬT ĐO ĐƯỢC (2026-09-11, feature_columns như trên): GT1 xgboost=21.11% >
    baseline=18.43% (XGBoost MAPE TỆ HƠN baseline dù Gate 4b pass theo RMSE!); WT1
    xgboost=17.74% ≈ baseline=17.81% (gần như ngang nhau). Đây là bằng chứng RMSE và
    MAPE có thể mâu thuẫn — RMSE nhạy với sai số tuyệt đối (nổi bật ở vùng công suất
    cao), MAPE nhạy với sai số tương đối (nổi bật ở vùng công suất thấp, dù đã lọc
    <50kW). KHÔNG đổi gate_4b_pass theo phát hiện này (gate vẫn chốt theo RMSE như thiết
    kế ban đầu) — chỉ ghi nhận để người dùng hiểu rõ giới hạn thật của Gate 4b khi diễn
    giải "XGBoost tốt hơn baseline".

    xgboost và sklearn.metrics được import LAZY bên trong hàm (cùng lý do phòng hờ như
    detect_anomalies_ml() — tránh lỗi thư viện ML làm sập cả module).

    include_predictions=False THEO MẶC ĐỊNH (agent/tool schema KHÔNG bao giờ truyền
    True — xem agent/turbine_tool_wrappers.py::predict_power_for_agent) vì lý do TƯƠNG
    TỰ classify_turbine_status(): không trả nguyên văn dữ liệu từng dòng cho LLM (tốn
    token/chi phí vô ích). Chỉ dùng True khi gọi TRỰC TIẾP từ code nội bộ (Dashboard tab
    Power Curve) cần overlay predicted power lên biểu đồ — khi đó, VÀ CHỈ KHI
    method_used=="xgboost", trả thêm "predictions": {"row_index", wind_speed_column,
    "predicted_power"} CHỈ cho tập test (~20% cuối, KHÔNG phải toàn bộ dataset).
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        feature_columns = feature_columns or []
        all_features = [wind_speed_column] + [c for c in feature_columns if c != wind_speed_column]
        for col in [power_column] + all_features:
            col_error = _validate_column(manager, col, dataset_name)
            if col_error:
                return failure(col_error)
        leakage_risk_cols = [c for c in ("GT1.Common.TheoActiveW", "GT1.SP.ActualSPP") if c in all_features]
        if leakage_risk_cols:
            return failure(
                f"Không được dùng {leakage_risk_cols} làm feature — gây data leakage: "
                "TheoActiveW gần như là đáp án lý thuyết cho power_column; ActualSPP "
                "(setpoint) gần như quyết định trực tiếp power lúc Curtailment (xem rule "
                "Curtailment trong classify_turbine_status(): |power-setpoint|/setpoint < "
                "5%), nên cũng là leakage tương tự dù không dùng ~100% thời gian."
            )

        n = len(df)
        split_idx = int(n * train_fraction)
        class_series = pd.Series(classification)
        class_series.index = class_series.index.astype(df.index.dtype) if len(class_series) else class_series.index
        class_series = class_series.reindex(df.index)

        valid_mask = df[all_features + [power_column]].notna().all(axis=1)
        # dung VI TRI THU TU trong df (khong phai gia tri index) de xac dinh 80% dau/20%
        # cuoi — an toan ke ca khi index khong lien tuc tu 0
        row_position = pd.Series(range(n), index=df.index)
        train_mask = (row_position < split_idx) & (class_series == "Normal") & valid_mask
        test_mask = (row_position >= split_idx) & valid_mask

        def _fallback(reason: str) -> dict:
            try:
                baseline_pred_fb = _power_curve_interpolate(power_curve, df.loc[test_mask, wind_speed_column].values)
            except ValueError as exc:
                return failure(str(exc))
            actual_fb = df.loc[test_mask, power_column].values
            if len(actual_fb) == 0:
                return failure("Không có điểm nào trong tập test để đánh giá baseline (kiểm tra lại train_fraction/dữ liệu).")
            rmse_fb = float(np.sqrt(np.mean((actual_fb - baseline_pred_fb) ** 2)))
            mae_fb = float(np.mean(np.abs(actual_fb - baseline_pred_fb)))
            ss_res = float(np.sum((actual_fb - baseline_pred_fb) ** 2))
            ss_tot = float(np.sum((actual_fb - actual_fb.mean()) ** 2))
            r2_fb = 1 - ss_res / ss_tot if ss_tot > 0 else None
            mape_fb = _mape(actual_fb, baseline_pred_fb, MAPE_MIN_ACTUAL_POWER_KW)
            return success({
                "method_used": "power_curve_interpolation",
                "ml_gate_reason": reason,
                "test_size": int(len(actual_fb)),
                "metrics": {
                    "rmse": round(rmse_fb, 4), "mae": round(mae_fb, 4),
                    "r2": (round(r2_fb, 4) if r2_fb is not None else None),
                    "mape": (round(mape_fb, 4) if mape_fb is not None else None),
                },
            })

        if train_mask.sum() < 30:
            return _fallback(f"Không đủ điểm 'Normal' trong tập train ({int(train_mask.sum())}, cần >=30).")
        if test_mask.sum() < 10:
            return _fallback(f"Không đủ điểm trong tập test ({int(test_mask.sum())}, cần >=10).")

        try:
            import xgboost as xgb
        except ImportError as exc:
            return _fallback(f"Không import được xgboost ({exc}) — dùng fallback power curve interpolation.")

        X_train, y_train = df.loc[train_mask, all_features], df.loc[train_mask, power_column]
        X_test, y_test = df.loc[test_mask, all_features], df.loc[test_mask, power_column]

        model = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_test)

        baseline_pred = _power_curve_interpolate(power_curve, df.loc[test_mask, wind_speed_column].values)

        y_test_arr = y_test.values
        rmse_xgb = float(np.sqrt(np.mean((y_test_arr - y_pred_xgb) ** 2)))
        rmse_baseline = float(np.sqrt(np.mean((y_test_arr - baseline_pred) ** 2)))
        mae_xgb = float(np.mean(np.abs(y_test_arr - y_pred_xgb)))
        mae_baseline = float(np.mean(np.abs(y_test_arr - baseline_pred)))

        def _r2(actual, pred):
            ss_res = float(np.sum((actual - pred) ** 2))
            ss_tot = float(np.sum((actual - actual.mean()) ** 2))
            return (1 - ss_res / ss_tot) if ss_tot > 0 else None

        r2_xgb = _r2(y_test_arr, y_pred_xgb)
        r2_baseline = _r2(y_test_arr, baseline_pred)
        # MAPE CHI de hien thi them thong tin — KHONG dua vao gate 4b/4c (van chot theo
        # RMSE/feature importance nhu truoc, xem yeu cau task khong doi tieu chi gate).
        mape_xgb = _mape(y_test_arr, y_pred_xgb, MAPE_MIN_ACTUAL_POWER_KW)
        mape_baseline = _mape(y_test_arr, baseline_pred, MAPE_MIN_ACTUAL_POWER_KW)

        gate_4b_pass = bool(rmse_xgb < rmse_baseline)
        gate_4b_info = {
            "xgboost_rmse": round(rmse_xgb, 4),
            "baseline_rmse": round(rmse_baseline, 4),
            "xgboost_mae": round(mae_xgb, 4),
            "baseline_mae": round(mae_baseline, 4),
            "xgboost_r2": (round(r2_xgb, 4) if r2_xgb is not None else None),
            "baseline_r2": (round(r2_baseline, 4) if r2_baseline is not None else None),
            "gate_4b_pass": gate_4b_pass,
        }

        importances = model.feature_importances_
        ranking = [{"feature": f, "importance": round(float(imp), 6)} for f, imp in sorted(zip(all_features, importances), key=lambda x: -x[1])]
        wind_speed_rank = next(i for i, r in enumerate(ranking, 1) if r["feature"] == wind_speed_column)
        gate_4c_pass = bool(wind_speed_rank <= 2)
        gate_4c_info = {
            "feature_importance_ranking": ranking,
            "wind_speed_rank": wind_speed_rank,
            "gate_4c_pass": gate_4c_pass,
        }

        if not gate_4b_pass:
            result = _fallback(f"Gate 4b FAIL: XGBoost RMSE={rmse_xgb:.4f} không thấp hơn baseline RMSE={rmse_baseline:.4f}.")
            if result["ok"]:
                result["data"]["gate_4b"] = gate_4b_info
            return result

        if not gate_4c_pass:
            result = _fallback(
                f"Gate 4c FAIL: wind_speed xếp hạng #{wind_speed_rank} (cần <=2) — nghi ngờ "
                "bug preprocessing (feature lệch thời gian, scale sai...), KHÔNG dùng XGBoost "
                "dù Gate 4b có pass."
            )
            if result["ok"]:
                result["data"]["gate_4b"] = gate_4b_info
                result["data"]["gate_4c"] = gate_4c_info
            return result

        result_data = {
            "method_used": "xgboost",
            "gate_4b": gate_4b_info,
            "gate_4c": gate_4c_info,
            "feature_columns": all_features,
            "test_size": int(len(y_test_arr)),
            "metrics": {
                "rmse": round(rmse_xgb, 4), "mae": round(mae_xgb, 4),
                "r2": (round(r2_xgb, 4) if r2_xgb is not None else None),
                "mape": (round(mape_xgb, 4) if mape_xgb is not None else None),
            },
            "baseline_metrics": {
                "rmse": round(rmse_baseline, 4), "mae": round(mae_baseline, 4),
                "r2": (round(r2_baseline, 4) if r2_baseline is not None else None),
                "mape": (round(mape_baseline, 4) if mape_baseline is not None else None),
            },
        }
        if include_predictions:
            result_data["predictions"] = {
                "row_index": df.loc[test_mask].index.tolist(),
                wind_speed_column: df.loc[test_mask, wind_speed_column].tolist(),
                "predicted_power": [round(float(v), 4) for v in y_pred_xgb],
            }
        return success(result_data)
    except Exception as exc:
        return failure(str(exc))


# Nhan bieu thi turbine THUC SU DUNG (khong san xuat) trong 8 nhan Meteodyn — DUY NHAT
# 2 nhan nay co nghia "cong suat gan 0" theo dinh nghia rule (xem CLASSIFICATION_LABELS
# va docstring classify_turbine_status() o tren). "Measurement Error" KHONG duoc xep
# vao available/unavailable — du lieu cam bien loi nen KHONG BIET turbine thuc su chay
# hay dung, dua vao mau so se suy dien sai tu du lieu khong dang tin (loai hoan toan
# khoi tinh TBA, khong tinh la available lan unavailable). Day la GIA DINH HEURISTIC
# cua module nay — KHONG phai cong thuc trich tu tai lieu PO024 (da xac nhan qua tra
# cuu KB that: PO024 chi liet ke 6 nhan phan loai dung trong methodology, KHONG neu ro
# quy tac nhan nao tinh la "available" cho TBA/PBA).
_TBA_UNAVAILABLE_LABELS = {"Stop", "Partial Stop"}
_TBA_EXCLUDED_LABELS = {"Measurement Error"}

# Disclaimer đi kèm tba_pct trong output calculate_turbine_kpis() — TBA ở đây là XẤP XỈ
# MVP dựa trên % thời gian KHÔNG ở trạng thái Stop/Partial Stop (xem 2 constant ở trên),
# KHÔNG PHẢI TBA chuẩn hợp đồng (contractual TBA) — TBA chuẩn cần tách riêng nguyên nhân
# "not producing" (low-wind = turbine sẵn sàng nhưng không đủ gió, KHÔNG tính là
# unavailable trong TBA hợp đồng thật; grid-outage = lỗi lưới điện bên ngoài, thường
# LOẠI KHỎI TBA vì không phải lỗi turbine; planned-maintenance = bảo trì theo kế hoạch,
# thường tính riêng khỏi unplanned downtime) — rule Stop/Partial Stop hiện tại GỘP CHUNG
# tất cả nguyên nhân "công suất gần 0" vào 1 nhãn, không phân biệt được.
TBA_APPROXIMATION_NOTE = (
    "TBA xấp xỉ theo tỷ lệ thời gian ở trạng thái Power Production — chưa tách riêng "
    "low-wind/grid-outage/planned-maintenance, không phải TBA chuẩn hợp đồng."
)


def _infer_interval_hours(df: pd.DataFrame, interval_minutes: float | None) -> tuple[float, float]:
    """Suy ra interval_minutes tu mode() cua datetime.diff() neu khong truyen tay — TAI
    DUNG boi calculate_turbine_kpis() VA calculate_loss_by_category() (refactor tu logic
    goc chi co trong calculate_turbine_kpis() truoc day) de dam bao 2 ham LUON dung
    CHINH XAC cung interval_hours khi goi cung tham so — tranh lech ket qua do 2 noi tu
    suy interval khac nhau (vd 1 noi lam tron khac, hoac dataset co gap bat thuong khien
    mode() nhay cam thu tu tinh). Raise ValueError (KHONG failure() truc tiep) de caller
    tu quyet dinh cach bao loi, giong quy uoc _power_curve_interpolate()/
    _compute_reachable_power_per_row() da dung."""
    if "datetime" not in df.columns:
        raise ValueError(
            "Dataset không có cột 'datetime' đã derive (xem "
            "DatasetManager.derive_time_columns) — không thể tính interval theo thời gian."
        )
    if interval_minutes is None:
        diffs = df["datetime"].diff().dropna()
        mode_result = diffs.mode()
        if len(mode_result) == 0:
            raise ValueError(
                "Không suy ra được interval từ cột datetime (dữ liệu quá ít hoặc không "
                "có bản ghi liên tiếp)."
            )
        interval_minutes = mode_result.iloc[0].total_seconds() / 60
    return interval_minutes, interval_minutes / 60


def _compute_reachable_power_per_row(
    df: pd.DataFrame,
    theoretical_power_column: str | None,
    power_curve: dict | None,
    wind_speed_column: str | None,
) -> tuple[pd.Series, str]:
    """Tinh reachable_power (cong suat 'le ra dat duoc') moi dong — TAI DUNG boi
    calculate_turbine_kpis() VA calculate_loss_by_category() (refactor tu logic goc chi
    co trong calculate_turbine_kpis() truoc day), dam bao 2 ham KHONG BAO GIO tinh
    reachable khac nhau du goi voi cung tham so (neu khac nhau, tong loss_by_category se
    khong the doi chieu duoc voi losses_mwh mot cach dang tin cay).

    method="direct_theoretical" (Site 1, dung TRUC TIEP theoretical_power_column) hoac
    "estimated_power_curve" (Site 2, noi suy tuyen tinh tu power_curve qua
    _power_curve_interpolate() da co san — can wind_speed_column). Uu tien
    theoretical_power_column neu ca 2 deu duoc truyen (chinh xac hon uoc luong).
    Raise ValueError neu khong the tinh duoc (khong truyen gi ca, hoac power_curve
    khong hop le) — caller tu bat va return failure()."""
    if theoretical_power_column is not None:
        return df[theoretical_power_column].astype(float), "direct_theoretical"
    if power_curve is not None:
        power_curve_list = power_curve.get("power_curve")
        bin_width = power_curve.get("bin_width")
        if not power_curve_list or not bin_width:
            raise ValueError(
                "power_curve không hợp lệ — cần truyền field 'data' từ kết quả "
                "estimate_power_curve() (có 'power_curve' và 'bin_width'), không phải "
                "toàn bộ dict {'ok':..,'data':..}."
            )
        if wind_speed_column is None:
            raise ValueError(
                "Cần truyền wind_speed_column khi ước lượng reachable power qua "
                "power_curve (Site 2, không có theoretical_power_column)."
            )
        interpolated = _power_curve_interpolate(power_curve, df[wind_speed_column].astype(float).values)
        return pd.Series(interpolated, index=df.index), "estimated_power_curve"
    raise ValueError(
        "Cần truyền theoretical_power_column (Site 1) HOẶC power_curve (Site 2) để tính "
        "reachable energy — không có cả 2 thì không thể ước lượng tổn thất."
    )


def calculate_turbine_kpis(
    manager: DatasetManager,
    dataset_name: str,
    power_column: str,
    wind_speed_column: str,
    classification: dict,
    power_curve: dict | None = None,
    theoretical_power_column: str | None = None,
    interval_minutes: float | None = None,
) -> dict:
    """
    Tinh KPI theo dung format bang PO024 (Production/Losses/Capacity Factor/TBA/PBA).
    TAI DUNG HOAN TOAN classification (tu classify_turbine_status(), KHONG phan loai
    lai) va power_curve (tu estimate_power_curve(), KHONG uoc luong lai) — day chi la
    lop TONG HOP tren ket qua 2 tool do da co san.

    2 cach tinh "reachable energy" (san luong LE RA dat duoc neu khong co ton that),
    tuy site:
    - method="direct_theoretical" (Site 1, GT1.Common.TheoActiveW co san): dung TRUC
      TIEP cot ly thuyet, KHONG can power_curve. Truyen theoretical_power_column.
    - method="estimated_power_curve" (Site 2, khong co cot ly thuyet): uoc luong cong
      suat "le ra dat duoc" tai moi diem bang noi suy tuyen tinh giua cac median_power
      trong power_curve — TAI DUNG _power_curve_interpolate() da co san (dung boi
      predict_power() lam baseline), KHONG viet lai logic binning.

    Neu CA HAI deu duoc truyen (vd goi tay muon so sanh), uu tien theoretical_power_column
    (do chinh xac hon uoc luong tu power curve).

    GIA DINH HEURISTIC (can biet truoc khi tin ket qua — xem _TBA_UNAVAILABLE_LABELS o
    tren de biet ro day la quyet dinh cua module nay, KHONG phai trich tu PO024):
    - power_column/theoretical_power_column don vi kW (da verify that qua du lieu GT1/
      WT1: GT1.Grid.totW max~4525, GT1.Common.TheoActiveW max~4517 hop ly cho turbine
      ~4.5MW; WT1.ACTIVE_POWER max~2051 hop ly turbine ~2MW-class — KHONG phai MW).
    - TBA = % thoi gian KHONG o nhan "Stop"/"Partial Stop" (2 nhan duy nhat bieu thi
      cong suat gan 0 do turbine dung). "Measurement Error" (du lieu cam bien loi, KHONG
      biet turbine thuc su chay hay dung) bi LOAI KHOI mau so hoan toan. Curtailment/
      Partial Curtailment tinh la AVAILABLE (turbine co kha nang chay, chi bi gioi han
      theo lenh dieu do, khong phai loi turbine). ĐÂY LÀ XẤP XỈ MVP, KHÔNG PHẢI TBA
      chuẩn hợp đồng — chưa tách riêng low-wind (turbine sẵn sàng nhưng không đủ gió,
      TBA hợp đồng thường KHÔNG tính là unavailable)/grid-outage (lỗi lưới ngoài, thường
      loại khỏi TBA)/planned-maintenance (bảo trì theo kế hoạch, thường tính riêng) — cả
      3 nguyên nhân này hiện bị GỘP CHUNG vào rule "công suất gần 0" (xem field
      "tba_note" trong kết quả trả về, luôn kèm disclaimer này).
    - rated_power_kw: neu co power_curve, dung max(median_power) qua cac bin (CUNG cong
      thuc classify_turbine_status() dung de uoc luong rated_power_estimate — tai dung
      y tuong, khong tinh khac di). Neu KHONG co power_curve (chi co
      theoretical_power_column, method=direct), fallback max(power_column)*1.02 (bien
      an toan 2% — tranh danh gia capacity factor cao gia tao do rated_power uoc luong
      thap hon that).
    - production_mwh/reachable_mwh CHI tinh tren cac dong co reachable-power hop le (vd
      wind_speed roi vao 1 bin co du lieu that trong power_curve) — xem
      n_rows_excluded_no_reachable_estimate de biet co bao nhieu dong bi loai, neu con
      so nay lon so voi tong so dong thi capacity_factor_pct co the bi UNDERESTIMATE
      nhe (production thuc te cao hon phan da tinh, do vai dong bi loai khoi tong).

    Tra ve: production_mwh, reachable_mwh, losses_mwh (= reachable_mwh - production_mwh,
    THEM 2026-09-11 cho calculate_loss_by_category() doi chieu — xem docstring ham do),
    losses_pct, capacity_factor_pct, tba_pct, tba_note (disclaimer co dinh giai thich TBA
    la xap xi, xem TBA_APPROXIMATION_NOTE), pba_pct, rated_power_kw, method,
    interval_minutes, n_rows_used, n_rows_excluded_no_reachable_estimate.
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        for col in (power_column, wind_speed_column):
            col_error = _validate_column(manager, col, dataset_name)
            if col_error:
                return failure(col_error)

        if theoretical_power_column is not None:
            col_error = _validate_column(manager, theoretical_power_column, dataset_name)
            if col_error:
                return failure(col_error)

        try:
            interval_minutes, interval_hours = _infer_interval_hours(df, interval_minutes)
        except ValueError as exc:
            return failure(str(exc))

        # Validate + xay median_lookup TRUOC (giu nguyen thu tu goc) — dung cho
        # rated_power_kw ben duoi BAT KE method thuc su dung la gi (vd theoretical_power_
        # column duoc uu tien cho reachable, nhung power_curve VAN duoc validate/dung de
        # uoc luong rated_power neu co truyen kem).
        median_lookup = None
        if power_curve is not None:
            power_curve_list = power_curve.get("power_curve")
            bin_width = power_curve.get("bin_width")
            if not power_curve_list or not bin_width:
                return failure(
                    "power_curve không hợp lệ — cần truyền field 'data' từ kết quả "
                    "estimate_power_curve() (có 'power_curve' và 'bin_width'), không phải "
                    "toàn bộ dict {'ok':..,'data':..}."
                )
            median_lookup = pd.Series({b["wind_speed_bin_center"]: b["median_power"] for b in power_curve_list})

        try:
            reachable_per_row, method = _compute_reachable_power_per_row(
                df, theoretical_power_column, power_curve, wind_speed_column,
            )
        except ValueError as exc:
            return failure(str(exc))

        actual_per_row = df[power_column].astype(float)

        valid_mask = reachable_per_row.notna() & actual_per_row.notna()
        n_rows_used = int(valid_mask.sum())
        n_rows_excluded = int((~valid_mask).sum())
        if n_rows_used == 0:
            return failure("Không có dòng nào có đủ dữ liệu (power + reachable) để tính KPI.")

        actual_valid = actual_per_row[valid_mask]
        reachable_valid = reachable_per_row[valid_mask]

        # kWh = kW * gio, /1000 -> MWh. Gia tri am (parasitic load, xem module docstring)
        # GIU NGUYEN trong tong (khong loc bo) - phan anh dung san luong THUC.
        production_mwh = float(actual_valid.sum() * interval_hours / 1000)
        reachable_mwh = float(reachable_valid.sum() * interval_hours / 1000)

        if reachable_mwh > 0:
            pba_pct = round(production_mwh / reachable_mwh * 100, 4)
            losses_pct = round(100 - pba_pct, 4)
        else:
            pba_pct = None
            losses_pct = None

        if median_lookup is not None:
            positive_medians = median_lookup.dropna()
            positive_medians = positive_medians[positive_medians > 0]
            rated_power_kw = (
                float(positive_medians.max()) if len(positive_medians) > 0
                else float(actual_per_row.max() * 1.02)
            )
        else:
            rated_power_kw = float(actual_per_row.max() * 1.02)

        total_hours = len(df) * interval_hours
        capacity_factor_pct = (
            round(production_mwh / (rated_power_kw / 1000 * total_hours) * 100, 4)
            if rated_power_kw > 0 and total_hours > 0 else None
        )

        labels = pd.Series(classification)
        counted = labels[~labels.isin(_TBA_EXCLUDED_LABELS)]
        tba_pct = (
            round((~counted.isin(_TBA_UNAVAILABLE_LABELS)).sum() / len(counted) * 100, 4)
            if len(counted) > 0 else None
        )

        return success({
            "production_mwh": round(production_mwh, 4),
            "reachable_mwh": round(reachable_mwh, 4),
            "losses_mwh": round(reachable_mwh - production_mwh, 4),
            "losses_pct": losses_pct,
            "capacity_factor_pct": capacity_factor_pct,
            "tba_pct": tba_pct,
            "tba_note": TBA_APPROXIMATION_NOTE,
            "pba_pct": pba_pct,
            "rated_power_kw": round(rated_power_kw, 4),
            "method": method,
            "interval_minutes": interval_minutes,
            "n_rows_used": n_rows_used,
            "n_rows_excluded_no_reachable_estimate": n_rows_excluded,
        })
    except Exception as exc:
        return failure(str(exc))


def calculate_loss_by_category(
    manager: DatasetManager,
    dataset_name: str,
    power_column: str,
    wind_speed_column: str,
    classification: dict,
    power_curve: dict | None = None,
    theoretical_power_column: str | None = None,
    interval_minutes: float | None = None,
) -> dict:
    """
    Năng lượng lãng phí (MWh) THEO TỪNG NHÃN trong 8 nhãn Meteodyn — nền tảng cho donut
    chart (Dashboard Tab 1) và farm-level ranking (Tab 0). Khác classify_turbine_status()
    (chỉ % THỜI GIAN mỗi nhãn) và calculate_turbine_kpis() (chỉ 1 con số losses_mwh TỔNG,
    không tách theo nhãn) — hàm này trả loss_mwh RIÊNG cho từng nhãn KHÔNG PHẢI "Normal".

    CÔNG THỨC (mỗi dòng KHÔNG phải "Normal"):
        loss_kwh = (reachable_power - actual_power) * interval_hours
    Tổng hợp theo nhãn: {label: sum(loss_kwh)/1000} (MWh). TÁI DÙNG HOÀN TOÀN
    _compute_reachable_power_per_row()/_infer_interval_hours() — CHÍNH XÁC cùng logic
    calculate_turbine_kpis() dùng (2 hàm private này được refactor RA từ
    calculate_turbine_kpis() để đảm bảo KHÔNG BAO GIỜ tính reachable/interval khác nhau
    giữa 2 tool khi gọi cùng tham số).

    wind_speed_column BẮT BUỘC (khác signature đề xuất ban đầu không liệt kê tham số
    này) — cần thiết để nội suy reachable power qua power_curve ở Site 2 (không có
    theoretical_power_column), giống hệt calculate_turbine_kpis() đã yêu cầu.

    QUAN TRỌNG — tại sao tổng loss_by_category_mwh KHÔNG khớp losses_mwh của
    calculate_turbine_kpis() (ĐÃ ĐO THẬT 2026-09-11, KHÔNG PHẢI giả định "sai số nhỏ" —
    giả thuyết ban đầu ĐÃ SAI, xem lại bên dưới): losses_mwh (calculate_turbine_kpis) =
    reachable_mwh - production_mwh tính trên TOÀN BỘ dòng hợp lệ (kể cả nhãn "Normal"),
    còn hàm này CHỈ cộng dồn trên dòng KHÔNG PHẢI "Normal" (đúng công thức được yêu cầu).
    Vì tổng-toàn-bộ = tổng(Normal) + tổng(non-Normal) MỘT CÁCH TUYỆT ĐỐI (đẳng thức toán
    học, không phải xấp xỉ), nên:
        losses_mwh(KPI) - total_loss_mwh(hàm này) == tổng (reachable-actual) CHỈ trên
        dòng "Normal" — ĐÃ VERIFY KHỚP CHÍNH XÁC đến 4 chữ số thập phân trên cả GT1/WT1
        (script kiểm chứng riêng, không qua 2 hàm này để tránh false-positive do dùng
        chung 1 lỗi).

    Ban đầu dự đoán SAI rằng khoảng chênh này "nhỏ, gần 0" (residual Normal đối xứng
    quanh trung vị nên trung hoà) — ĐO THẬT cho thấy KHÔNG nhỏ và dấu KHÔNG nhất quán:
    - GT1 (direct_theoretical): dòng Normal đóng góp +177.77 MWh = 17.6% tổng losses_mwh
      (1010.37 MWh). Nguyên nhân: reachable_power = TheoActiveW (mô hình lý thuyết) —
      MỘT tham chiếu HOÀN TOÀN KHÁC với power-curve median (chuẩn dùng để phân loại
      "Normal"). Không có lý do cấu trúc nào để 2 đại lượng khác nhau này triệt tiêu khi
      cộng dồn trên tập "Normal" (theo power curve) — chỉ là ~1 kWh/dòng nhưng nhân với
      180.499 dòng Normal (77% dữ liệu) thành số MWh đáng kể.
    - WT1 (estimated_power_curve): dòng Normal đóng góp -111.48 MWh = -38.7% tổng
      losses_mwh (287.84 MWh) — NGƯỢC DẤU với GT1 dù reachable ở đây CHÍNH LÀ power-curve
      median (cùng tham chiếu dùng phân loại Normal). Giả thuyết: sau khi loại 2 đuôi cực
      trị (Under/Over Production, ±2σ) khỏi phân phối residual, phần "lõi" còn lại (chính
      là "Normal") không còn đối xứng quanh trung vị gốc — mean-của-phần-còn-lại lệch nhẹ
      khỏi median-của-toàn-bộ. CHƯA verify sâu hơn (vd histogram residual Normal) — đây
      là giả thuyết hợp lý, không phải kết luận đã kiểm chứng.
    KẾT LUẬN: loss_by_category_mwh là 1 breakdown THEO ĐÚNG CÔNG THỨC được yêu cầu (chỉ
    nguyên-nhân-bất-thường, không gồm biến thiên "Normal" bình thường) — KHÔNG PHẢI, và
    KHÔNG NÊN kỳ vọng là, một phép chia lại 100% của losses_mwh. Muốn dùng cho mục đích
    đối chiếu tổng số tuyệt đối, phải cộng thêm phần "Normal" (không có trong dict trả
    về) — xem "Trả về" bên dưới.

    Trả về: {"loss_by_category_mwh": {label: mwh, ...} (CHỈ nhãn không-Normal xuất hiện
    trong classification), "total_loss_mwh": tổng tất cả nhãn trên, "interval_minutes",
    "method", "n_rows_used"}.
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        for col in (power_column, wind_speed_column):
            col_error = _validate_column(manager, col, dataset_name)
            if col_error:
                return failure(col_error)
        if theoretical_power_column is not None:
            col_error = _validate_column(manager, theoretical_power_column, dataset_name)
            if col_error:
                return failure(col_error)

        try:
            interval_minutes, interval_hours = _infer_interval_hours(df, interval_minutes)
        except ValueError as exc:
            return failure(str(exc))

        try:
            reachable_per_row, method = _compute_reachable_power_per_row(
                df, theoretical_power_column, power_curve, wind_speed_column,
            )
        except ValueError as exc:
            return failure(str(exc))

        actual_per_row = df[power_column].astype(float)
        valid_mask = reachable_per_row.notna() & actual_per_row.notna()
        n_rows_used = int(valid_mask.sum())
        if n_rows_used == 0:
            return failure("Không có dòng nào có đủ dữ liệu (power + reachable) để tính loss theo nhãn.")

        labels = pd.Series(classification).reindex(df.index)
        non_normal_mask = valid_mask & labels.notna() & (labels != "Normal")

        loss_mwh_per_row = (reachable_per_row - actual_per_row) * interval_hours / 1000
        grouped_sum = loss_mwh_per_row[non_normal_mask].groupby(labels[non_normal_mask]).sum()
        loss_by_category_mwh = {label: round(float(total), 4) for label, total in grouped_sum.items()}
        total_loss_mwh = round(float(sum(loss_by_category_mwh.values())), 4)

        return success({
            "loss_by_category_mwh": loss_by_category_mwh,
            "total_loss_mwh": total_loss_mwh,
            "interval_minutes": interval_minutes,
            "method": method,
            "n_rows_used": n_rows_used,
        })
    except Exception as exc:
        return failure(str(exc))
