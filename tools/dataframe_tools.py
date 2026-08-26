import pandas as pd

from tools.dataset_manager import DatasetManager
from tools.tool_results import failure, success

AGGREGATION_WHITELIST = ["mean", "sum", "min", "max", "median", "std", "count"]
GROUP_BY_WHITELIST = ["hour", "day", "month", "weekday", "year", "date"]


def _validate_column(manager: DatasetManager, column: str) -> str | None:
    columns = manager.get_columns()
    if column not in columns:
        return f"Column '{column}' not found in dataset. Valid columns are: {columns}"
    return None


def _validate_aggregation(aggregation: str) -> str | None:
    if aggregation not in AGGREGATION_WHITELIST:
        return f"Aggregation '{aggregation}' not allowed. Valid aggregations are: {AGGREGATION_WHITELIST}"
    return None


def _to_json_number(value):
    return float(value) if pd.notna(value) else None


def get_dataset_summary(manager: DatasetManager) -> dict:
    try:
        df = manager.get_dataframe()
        columns = manager.get_columns()
        numeric_columns = manager.get_numeric_columns()

        if "datetime" in df.columns:
            date_range = {"min": str(df["datetime"].min()), "max": str(df["datetime"].max())}
        else:
            date_range = None

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return success(
            {
                "row_count": len(df),
                "columns": columns,
                "numeric_columns": numeric_columns,
                "date_range": date_range,
                "dtypes": dtypes,
            }
        )
    except Exception as exc:
        return failure(str(exc))


def aggregate_metric(
    manager: DatasetManager,
    column: str,
    aggregation: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    try:
        col_error = _validate_column(manager, column)
        if col_error:
            return failure(col_error)

        agg_error = _validate_aggregation(aggregation)
        if agg_error:
            return failure(agg_error)

        df = manager.get_dataframe()
        date_range_applied = {"start_date": start_date, "end_date": end_date}

        filtered = df
        if start_date is not None or end_date is not None:
            if "datetime" not in df.columns:
                return failure("Dataset has no 'datetime' column; cannot filter by date.")
            if start_date is not None:
                try:
                    start_ts = pd.to_datetime(start_date)
                except (ValueError, TypeError):
                    return failure(f"Invalid start_date format: '{start_date}'")
                filtered = filtered[filtered["datetime"] >= start_ts]
            if end_date is not None:
                try:
                    end_ts = pd.to_datetime(end_date)
                except (ValueError, TypeError):
                    return failure(f"Invalid end_date format: '{end_date}'")
                filtered = filtered[filtered["datetime"] <= end_ts]

        if len(filtered) == 0:
            return failure(
                f"No rows found for the given date range (start_date={start_date}, end_date={end_date})."
            )

        value = getattr(filtered[column], aggregation)()

        return success(
            {
                "value": _to_json_number(value),
                "column": column,
                "aggregation": aggregation,
                "rows_used": len(filtered),
                "date_range_applied": date_range_applied,
            }
        )
    except Exception as exc:
        return failure(str(exc))


def group_metric(manager: DatasetManager, column: str, aggregation: str, group_by: str) -> dict:
    try:
        col_error = _validate_column(manager, column)
        if col_error:
            return failure(col_error)

        agg_error = _validate_aggregation(aggregation)
        if agg_error:
            return failure(agg_error)

        if group_by not in GROUP_BY_WHITELIST:
            return failure(f"group_by '{group_by}' not allowed. Valid group_by values are: {GROUP_BY_WHITELIST}")

        df = manager.get_dataframe()
        if group_by not in df.columns:
            return failure(f"Column '{group_by}' not found in dataset. Valid columns are: {manager.get_columns()}")

        grouped = df.groupby(group_by)[column]
        result_series = getattr(grouped, aggregation)().sort_index()
        values = {str(key): _to_json_number(val) for key, val in result_series.items()}

        return success(
            {
                "values": values,
                "column": column,
                "aggregation": aggregation,
                "group_by": group_by,
            }
        )
    except Exception as exc:
        return failure(str(exc))


def find_peak(manager: DatasetManager, column: str, mode: str = "max") -> dict:
    try:
        col_error = _validate_column(manager, column)
        if col_error:
            return failure(col_error)

        if mode not in ("max", "min"):
            return failure(f"mode '{mode}' not allowed. Valid modes are: ['max', 'min']")

        df = manager.get_dataframe()
        if len(df) == 0:
            return failure("Dataset is empty.")

        idx = df[column].idxmax() if mode == "max" else df[column].idxmin()
        row = df.loc[idx]
        value = row[column]

        if "datetime" in df.columns and pd.notna(row["datetime"]):
            dt_str = pd.Timestamp(row["datetime"]).isoformat()
        else:
            dt_str = None

        return success(
            {
                "value": _to_json_number(value),
                "datetime": dt_str,
                "row_index": int(idx),
                "mode": mode,
                "column": column,
            }
        )
    except Exception as exc:
        return failure(str(exc))


def compare_periods(
    manager: DatasetManager,
    column: str,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
    aggregation: str = "mean",
) -> dict:
    try:
        col_error = _validate_column(manager, column)
        if col_error:
            return failure(col_error)

        agg_error = _validate_aggregation(aggregation)
        if agg_error:
            return failure(agg_error)

        df = manager.get_dataframe()
        if "datetime" not in df.columns:
            return failure("Dataset has no 'datetime' column; cannot filter by period.")

        def _filtered_period(start, end, label):
            try:
                start_ts = pd.to_datetime(start)
                end_ts = pd.to_datetime(end)
            except (ValueError, TypeError):
                return None, f"Invalid date format for {label}: start='{start}', end='{end}'"
            subset = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)]
            if len(subset) == 0:
                return None, f"No rows found for {label} (start={start}, end={end})."
            return subset, None

        subset_a, err_a = _filtered_period(period_a_start, period_a_end, "period_a")
        if err_a:
            return failure(err_a)

        subset_b, err_b = _filtered_period(period_b_start, period_b_end, "period_b")
        if err_b:
            return failure(err_b)

        value_a = _to_json_number(getattr(subset_a[column], aggregation)())
        value_b = _to_json_number(getattr(subset_b[column], aggregation)())

        if value_a is None or value_b is None:
            absolute_difference = None
            percent_change = None
            percent_change_note = "percent_change unavailable because one of the aggregated values is NaN."
        elif value_a == 0:
            absolute_difference = value_b - value_a
            percent_change = None
            percent_change_note = "percent_change unavailable because period_a value is 0 (division by zero)."
        else:
            absolute_difference = value_b - value_a
            percent_change = round((value_b - value_a) / abs(value_a) * 100, 4)
            percent_change_note = None

        return success(
            {
                "period_a": {
                    "value": value_a,
                    "rows_used": len(subset_a),
                    "start": period_a_start,
                    "end": period_a_end,
                },
                "period_b": {
                    "value": value_b,
                    "rows_used": len(subset_b),
                    "start": period_b_start,
                    "end": period_b_end,
                },
                "absolute_difference": absolute_difference,
                "percent_change": percent_change,
                "percent_change_note": percent_change_note,
            }
        )
    except Exception as exc:
        return failure(str(exc))


def calculate_correlation(
    manager: DatasetManager,
    column_a: str,
    column_b: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    try:
        col_a_error = _validate_column(manager, column_a)
        if col_a_error:
            return failure(col_a_error)

        col_b_error = _validate_column(manager, column_b)
        if col_b_error:
            return failure(col_b_error)

        df = manager.get_dataframe()
        filtered = df

        if start_date is not None or end_date is not None:
            if "datetime" not in df.columns:
                return failure("Dataset has no 'datetime' column; cannot filter by date.")
            if start_date is not None:
                try:
                    start_ts = pd.to_datetime(start_date)
                except (ValueError, TypeError):
                    return failure(f"Invalid start_date format: '{start_date}'")
                filtered = filtered[filtered["datetime"] >= start_ts]
            if end_date is not None:
                try:
                    end_ts = pd.to_datetime(end_date)
                except (ValueError, TypeError):
                    return failure(f"Invalid end_date format: '{end_date}'")
                filtered = filtered[filtered["datetime"] <= end_ts]

        if len(filtered) == 0:
            return failure(
                f"No rows found for the given date range (start_date={start_date}, end_date={end_date})."
            )

        corr = filtered[column_a].corr(filtered[column_b])
        if pd.isna(corr):
            return failure(
                f"Correlation could not be computed for columns '{column_a}' and '{column_b}' "
                "(insufficient variance or data)."
            )

        corr_rounded = round(float(corr), 4)

        if corr_rounded > 0.7:
            interpretation = "strong positive"
        elif corr_rounded >= 0.3:
            interpretation = "moderate positive"
        elif corr_rounded >= -0.3:
            interpretation = "weak/no correlation"
        elif corr_rounded >= -0.7:
            interpretation = "moderate negative"
        else:
            interpretation = "strong negative"

        return success(
            {
                "correlation_coefficient": corr_rounded,
                "column_a": column_a,
                "column_b": column_b,
                "rows_used": len(filtered),
                "interpretation": interpretation,
            }
        )
    except Exception as exc:
        return failure(str(exc))
