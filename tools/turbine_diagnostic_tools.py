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

        def _numeric_compare(col):
            if col not in df.columns:
                return None
            return {
                "violating": {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in violating[col].describe().items()},
                "normal": {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in normal[col].describe().items()},
            }

        def _categorical_compare(col):
            if col not in df.columns:
                return None
            v_counts = violating[col].value_counts(normalize=True).mul(100).round(4)
            n_counts = normal[col].value_counts(normalize=True).mul(100).round(4)
            return {
                "violating_pct": {str(k): float(v) for k, v in v_counts.items()},
                "normal_pct": {str(k): float(v) for k, v in n_counts.items()},
            }

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
