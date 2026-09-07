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
        rules = {}

        # Rule 1: power am - magnitude-aware, KHONG dung nguong tuyet doi "power < 0"
        positive_power = df.loc[df[power_col] > 0, power_col]
        if len(positive_power) == 0:
            rules["negative_power_magnitude_aware"] = {
                "applicable": False,
                "reason": f"Không có giá trị {power_col} dương nào để ước lượng rated power.",
            }
        else:
            rated_power = float(np.percentile(positive_power, 99.5))
            neg_threshold = -NEGATIVE_POWER_RATED_FRACTION * rated_power
            mask = df[power_col] < neg_threshold
            rules["negative_power_magnitude_aware"] = {
                "applicable": True,
                "rated_power_estimate": round(rated_power, 4),
                "threshold_used": round(neg_threshold, 4),
                "count": int(mask.sum()),
                "pct": round(100 * mask.sum() / n, 4),
                "examples": _examples_from_mask(df, mask, ["datetime", power_col, wind_col]),
            }

        # Rule 2: power duong khi gio duoi cut-in
        mask = (df[power_col] > 0) & (df[wind_col] < CUT_IN_SPEED_MS)
        rules["power_below_cutin"] = {
            "applicable": True,
            "cutin_speed_ms": CUT_IN_SPEED_MS,
            "count": int(mask.sum()),
            "pct": round(100 * mask.sum() / n, 4),
            "examples": _examples_from_mask(df, mask, ["datetime", power_col, wind_col]),
        }

        # Rule 3: power duong khi gio tren cut-out
        mask = (df[power_col] > 0) & (df[wind_col] > CUT_OUT_SPEED_MS)
        rules["power_above_cutout"] = {
            "applicable": True,
            "cutout_speed_ms": CUT_OUT_SPEED_MS,
            "count": int(mask.sum()),
            "pct": round(100 * mask.sum() / n, 4),
            "examples": _examples_from_mask(df, mask, ["datetime", power_col, wind_col]),
        }

        # Rule 4: pitch 3 luoi lech >5 do - GT1 only
        pitch_cols = ["GT1.Pitch.Angle1", "GT1.Pitch.Angle2", "GT1.Pitch.Angle3"]
        if site == "site1" and all(c in df.columns for c in pitch_cols):
            pitch_max = df[pitch_cols].max(axis=1)
            pitch_min = df[pitch_cols].min(axis=1)
            mask = (pitch_max - pitch_min) > PITCH_MISMATCH_DEG
            rules["pitch_mismatch"] = {
                "applicable": True,
                "threshold_deg": PITCH_MISMATCH_DEG,
                "count": int(mask.sum()),
                "pct": round(100 * mask.sum() / n, 4),
                "examples": _examples_from_mask(df, mask, ["datetime"] + pitch_cols),
            }
        else:
            rules["pitch_mismatch"] = {"applicable": False, "reason": "Chỉ áp dụng cho Site 1 (GT1), cần đủ 3 cột GT1.Pitch.Angle1/2/3."}

        # Rule 5: 2 anemometer lech >20% - Site 2 only
        if site == "site2" and "WIND_SPEED_2" in df.columns:
            denom = df["WIND_SPEED"].replace(0, np.nan)
            pct_diff = (df["WIND_SPEED"] - df["WIND_SPEED_2"]).abs() / denom
            mask = pct_diff.fillna(0) > ANEMOMETER_MISMATCH_PCT
            rules["anemometer_mismatch"] = {
                "applicable": True,
                "threshold_pct": ANEMOMETER_MISMATCH_PCT,
                "count": int(mask.sum()),
                "pct": round(100 * mask.sum() / n, 4),
                "examples": _examples_from_mask(df, mask, ["datetime", "WIND_SPEED", "WIND_SPEED_2"]),
            }
        else:
            rules["anemometer_mismatch"] = {"applicable": False, "reason": "Chỉ áp dụng cho Site 2, cần cột WIND_SPEED_2."}

        # Rule 6: nhiet do vuot nguong - Site 2 only (khong tim thay threshold rieng trong
        # tai lieu KPI da ingest khi kiem tra - dung lai 90/150 nhu audit thu cong)
        temp_gearbox = "TEMPERATURE_GEARBOX_OIL"
        temp_stator = "TEMPERATURE_STATOR"
        if site == "site2" and temp_gearbox in df.columns and temp_stator in df.columns:
            mask = (df[temp_gearbox] > GEARBOX_OIL_ALARM_C) | (df[temp_stator] > STATOR_ALARM_C)
            rules["temperature_alarm"] = {
                "applicable": True,
                "gearbox_oil_threshold_c": GEARBOX_OIL_ALARM_C,
                "stator_threshold_c": STATOR_ALARM_C,
                "count": int(mask.sum()),
                "pct": round(100 * mask.sum() / n, 4),
                "examples": _examples_from_mask(df, mask, ["datetime", temp_gearbox, temp_stator]),
            }
        else:
            rules["temperature_alarm"] = {"applicable": False, "reason": "Chỉ áp dụng cho Site 2, cần cột TEMPERATURE_GEARBOX_OIL và TEMPERATURE_STATOR."}

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
