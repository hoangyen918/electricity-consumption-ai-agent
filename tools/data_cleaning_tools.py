"""
Tool cleaning tổng quát — dùng được cho BẤT KỲ dataset nào (không riêng turbine gió).
Theo đúng pattern tool cố định (KHÔNG code-execution tự do) như 6 tool Pandas hiện có
trong tools/dataframe_tools.py: nhận DatasetManager + dataset_name, trả dict qua
success()/failure() (tools/tool_results.py).

Các strategy áp dụng biến đổi (drop/ffill/interpolate) CHỈ trả về preview thống kê
(số dòng ảnh hưởng, before/after), KHÔNG ghi đè dataset gốc trong DatasetManager —
agent/user tự quyết định có áp dụng thật hay không (Phase 2 chỉ dừng ở mức chẩn đoán/
preview, chưa có cơ chế "apply" ghi lại dataset).
"""
import pandas as pd

from tools.dataset_manager import DatasetManager
from tools.tool_results import failure, success

MISSING_VALUE_STRATEGIES = ["report", "drop", "ffill", "interpolate"]
ANOMALY_METHODS = ["zscore", "iqr"]
MAX_INDEX_SAMPLE = 20


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


def clean_missing_values(
    manager: DatasetManager,
    dataset_name: str | None = None,
    column: str | None = None,
    strategy: str = "report",
) -> dict:
    """
    strategy="report": trả % missing theo từng cột (không đổi dữ liệu).
    strategy="drop": preview kết quả dropna() — nếu column=None, drop dòng có NaN ở
    BẤT KỲ cột nào (dropna() mặc định); nếu có column, chỉ drop dòng NaN ở đúng cột đó.
    strategy="ffill"/"interpolate": preview số giá trị sẽ được điền — nếu column=None,
    áp dụng cho toàn bộ dataframe; nếu có column, chỉ áp dụng cho đúng cột đó
    (interpolate chỉ áp dụng được cho cột số, sẽ báo lỗi rõ nếu cột không phải số).
    """
    try:
        if strategy not in MISSING_VALUE_STRATEGIES:
            return failure(f"Strategy '{strategy}' not allowed. Valid strategies: {MISSING_VALUE_STRATEGIES}")

        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        if column is not None:
            col_error = _validate_column(manager, column, dataset_name)
            if col_error:
                return failure(col_error)

        if strategy == "report":
            total_rows = len(df)
            missing_by_column = {}
            for col in df.columns:
                null_count = int(df[col].isnull().sum())
                if null_count > 0:
                    missing_by_column[col] = {
                        "count": null_count,
                        "pct": round(100 * null_count / total_rows, 4) if total_rows else 0.0,
                    }
            return success({
                "strategy": "report",
                "total_rows": total_rows,
                "missing_by_column": missing_by_column,
            })

        if strategy == "drop":
            rows_before = len(df)
            if column is not None:
                cleaned = df.dropna(subset=[column])
            else:
                cleaned = df.dropna()
            rows_after = len(cleaned)
            return success({
                "strategy": "drop",
                "column": column,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_dropped": rows_before - rows_after,
                "preview_note": "Kết quả preview, KHÔNG ghi đè dataset gốc trong DatasetManager.",
            })

        # ffill / interpolate
        if column is not None:
            if strategy == "interpolate" and not pd.api.types.is_numeric_dtype(df[column]):
                return failure(f"Column '{column}' không phải kiểu số, không thể interpolate().")
            before_null = int(df[column].isnull().sum())
            filled = df[column].ffill() if strategy == "ffill" else df[column].interpolate()
            after_null = int(filled.isnull().sum())
            values_filled = before_null - after_null
        else:
            before_null = int(df.isnull().sum().sum())
            if strategy == "ffill":
                filled_df = df.ffill()
            else:
                numeric_cols = df.select_dtypes(include="number").columns
                filled_df = df.copy()
                filled_df[numeric_cols] = filled_df[numeric_cols].interpolate()
            after_null = int(filled_df.isnull().sum().sum())
            values_filled = before_null - after_null

        return success({
            "strategy": strategy,
            "column": column,
            "values_filled": values_filled,
            "values_still_null_after": after_null,
            "preview_note": "Kết quả preview, KHÔNG ghi đè dataset gốc trong DatasetManager.",
        })
    except Exception as exc:
        return failure(str(exc))


def detect_duplicates(
    manager: DatasetManager,
    dataset_name: str | None = None,
    subset_cols: list[str] | None = None,
) -> dict:
    """
    Đếm duplicate rows theo pandas duplicated() mặc định (keep='first' — chỉ đếm các
    bản sao SAU bản gốc đầu tiên, đúng định nghĩa đã dùng ở audit thủ công
    `df.duplicated().sum()`). subset_cols=None -> so sánh toàn bộ dòng; có subset_cols
    -> chỉ so sánh các cột đó (vd chỉ theo timestamp để tìm trùng thời điểm ghi nhận
    dù giá trị cảm biến khác nhau).
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        if subset_cols is not None:
            missing_cols = [c for c in subset_cols if c not in df.columns]
            if missing_cols:
                return failure(f"Cột không tồn tại: {missing_cols}. Cột hợp lệ: {list(df.columns)}")

        dup_mask = df.duplicated(subset=subset_cols, keep="first")
        dup_count = int(dup_mask.sum())
        dup_indices = df.index[dup_mask].tolist()

        return success({
            "duplicate_count": dup_count,
            "subset_cols": subset_cols,
            "total_rows": len(df),
            "duplicate_row_indices_sample": [int(i) for i in dup_indices[:MAX_INDEX_SAMPLE]],
            "indices_truncated": len(dup_indices) > MAX_INDEX_SAMPLE,
        })
    except Exception as exc:
        return failure(str(exc))


def detect_statistical_anomalies(
    manager: DatasetManager,
    column: str,
    dataset_name: str | None = None,
    method: str = "zscore",
    threshold: float = 3.0,
) -> dict:
    """
    method="zscore": flag |value - mean| / std > threshold.
    method="iqr": flag ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (hệ số 1.5 cố định theo quy
    ước IQR chuẩn, threshold KHÔNG áp dụng cho method này).
    """
    try:
        if method not in ANOMALY_METHODS:
            return failure(f"Method '{method}' not allowed. Valid methods: {ANOMALY_METHODS}")

        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        col_error = _validate_column(manager, column, dataset_name)
        if col_error:
            return failure(col_error)

        if not pd.api.types.is_numeric_dtype(df[column]):
            return failure(f"Column '{column}' không phải kiểu số, không thể phát hiện anomaly.")

        series = df[column].dropna()
        if len(series) == 0:
            return failure(f"Column '{column}' không có giá trị hợp lệ (toàn bộ NaN).")

        if method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                return failure(f"Column '{column}' có std=0 hoặc không xác định, không thể tính z-score.")
            z = (series - mean) / std
            mask = z.abs() > threshold
            bounds = {"mean": round(float(mean), 6), "std": round(float(std), 6), "threshold": threshold}
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
            bounds = {"q1": round(float(q1), 6), "q3": round(float(q3), 6), "lower_bound": round(float(lower), 6), "upper_bound": round(float(upper), 6)}

        anomaly_indices = series.index[mask].tolist()
        anomaly_values = series[mask]

        return success({
            "method": method,
            "column": column,
            "anomaly_count": int(mask.sum()),
            "anomaly_pct": round(100 * mask.sum() / len(series), 4),
            "bounds": bounds,
            "anomaly_row_indices_sample": [int(i) for i in anomaly_indices[:MAX_INDEX_SAMPLE]],
            "anomaly_values_sample": [float(v) for v in anomaly_values.iloc[:MAX_INDEX_SAMPLE]],
            "indices_truncated": len(anomaly_indices) > MAX_INDEX_SAMPLE,
        })
    except Exception as exc:
        return failure(str(exc))


def correlation_analysis(
    manager: DatasetManager,
    target_column: str,
    dataset_name: str | None = None,
    top_n: int = 10,
) -> dict:
    """
    Correlation của target_column với TẤT CẢ cột số khác trong dataset (tự bỏ qua
    cột non-numeric và chính target_column). Sắp xếp theo |correlation| giảm dần
    nhưng GIỮ NGUYÊN dấu gốc trong output (không lấy trị tuyệt đối rồi mất dấu) —
    dùng cho bất kỳ dataset nào, không riêng turbine gió.
    """
    try:
        df = manager.get_dataframe(dataset_name)
        if df is None:
            return failure(_dataset_not_found(manager, dataset_name))

        col_error = _validate_column(manager, target_column, dataset_name)
        if col_error:
            return failure(col_error)

        if not pd.api.types.is_numeric_dtype(df[target_column]):
            return failure(f"Column '{target_column}' không phải kiểu số, không thể tính correlation.")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        other_numeric_cols = [c for c in numeric_cols if c != target_column]
        if not other_numeric_cols:
            return failure("Không có cột số nào khác để tính correlation.")

        correlations = []
        for col in other_numeric_cols:
            corr = df[target_column].corr(df[col])
            if pd.notna(corr):
                correlations.append({"column": col, "correlation": round(float(corr), 4)})

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return success({
            "target_column": target_column,
            "correlations": correlations[:top_n],
            "total_columns_compared": len(other_numeric_cols),
            "columns_returned": min(top_n, len(correlations)),
        })
    except Exception as exc:
        return failure(str(exc))
