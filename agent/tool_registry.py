"""
Tool registry: nhận tên tool + raw arguments (dict) từ LLM,
validate bằng Pydantic, rồi gọi đúng function thật.
Đây là lớp bảo vệ CHÍNH — không tin bất kỳ argument nào từ LLM
cho tới khi Pydantic xác nhận hợp lệ.
"""
from typing import Literal, Optional
from pydantic import BaseModel, ValidationError

from tools.dataset_manager import DatasetManager
from tools.dataframe_tools import (
    get_dataset_summary, aggregate_metric, group_metric,
    find_peak, compare_periods, calculate_correlation,
)
from tools.calculator_tools import calculate
from tools.data_cleaning_tools import (
    clean_missing_values, detect_duplicates, detect_statistical_anomalies, correlation_analysis,
)
from tools.turbine_diagnostic_tools import (
    detect_physical_violations, detect_sensor_issues, diagnose_theoretical_power_deviation,
    explain_anomaly,
)
from agent.turbine_tool_wrappers import (
    TURBINE_DEFAULT_BIN_WIDTH,
    estimate_power_curve_for_agent, classify_turbine_status_for_agent,
    group_anomaly_windows_for_agent, detect_anomalies_ml_for_agent, predict_power_for_agent,
    calculate_turbine_kpis_for_agent,
)
from rag.retriever import search_knowledge_base
from tools.tool_results import failure


AggregationType = Literal["mean", "sum", "min", "max", "median", "std", "count"]
GroupByType = Literal["hour", "day", "month", "weekday", "year", "date"]


class AggregateMetricArgs(BaseModel):
    column: str
    aggregation: AggregationType
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dataset_name: Optional[str] = None


class GroupMetricArgs(BaseModel):
    column: str
    aggregation: AggregationType
    group_by: GroupByType
    dataset_name: Optional[str] = None


class FindPeakArgs(BaseModel):
    column: str
    mode: Literal["max", "min"] = "max"
    dataset_name: Optional[str] = None


class ComparePeriodsArgs(BaseModel):
    column: str
    period_a_start: str
    period_a_end: str
    period_b_start: str
    period_b_end: str
    aggregation: AggregationType = "mean"
    dataset_name: Optional[str] = None


class CalculateCorrelationArgs(BaseModel):
    column_a: str
    column_b: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dataset_name: Optional[str] = None


class CalculateArgs(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide", "percentage_of", "percentage_change"]
    value_a: float
    value_b: float


class SearchKnowledgeBaseArgs(BaseModel):
    query: str
    top_k: int = 8
    language: Optional[str] = None
    document_type: Optional[str] = None


# --- Nhóm generic (data_cleaning_tools.py) — dùng cho MỌI dataset, không riêng turbine ---

class CleanMissingValuesArgs(BaseModel):
    column: Optional[str] = None
    strategy: Literal["report", "drop", "ffill", "interpolate"] = "report"
    dataset_name: Optional[str] = None


class DetectDuplicatesArgs(BaseModel):
    subset_cols: Optional[list[str]] = None
    dataset_name: Optional[str] = None


class DetectStatisticalAnomaliesArgs(BaseModel):
    column: str
    method: Literal["zscore", "iqr"] = "zscore"
    threshold: float = 3.0
    dataset_name: Optional[str] = None


class CorrelationAnalysisArgs(BaseModel):
    target_column: str
    top_n: int = 10
    dataset_name: Optional[str] = None


class ExplainAnomalyArgs(BaseModel):
    anomaly_mask_column: Optional[str] = None
    row_indices: Optional[list[int]] = None
    compare_columns: Optional[list[str]] = None
    dataset_name: Optional[str] = None


# --- Nhóm turbine-specific (tools/turbine_diagnostic_tools.py) — tự nhận diện site,
# chỉ áp dụng đúng cho dataset turbine gió (GT1.*/ACTIVE_POWER+WIND_SPEED) ---

class DetectPhysicalViolationsArgs(BaseModel):
    dataset_name: Optional[str] = None


class DetectSensorIssuesArgs(BaseModel):
    flatline_min_length: int = 6
    gap_multiplier: float = 3.0
    dataset_name: Optional[str] = None


class DiagnoseTheoreticalPowerDeviationArgs(BaseModel):
    threshold_pct: float = 10.0
    dataset_name: Optional[str] = None


class EstimatePowerCurveArgs(BaseModel):
    power_column: Optional[str] = None
    wind_speed_column: Optional[str] = None
    bin_width: float = TURBINE_DEFAULT_BIN_WIDTH
    dataset_name: Optional[str] = None


class ClassifyTurbineStatusArgs(BaseModel):
    power_column: Optional[str] = None
    wind_speed_column: Optional[str] = None
    setpoint_column: Optional[str] = None
    bin_width: float = TURBINE_DEFAULT_BIN_WIDTH
    dataset_name: Optional[str] = None


class GroupAnomalyWindowsArgs(BaseModel):
    timestamp_column: str = "datetime"
    power_column: Optional[str] = None
    wind_speed_column: Optional[str] = None
    setpoint_column: Optional[str] = None
    bin_width: float = TURBINE_DEFAULT_BIN_WIDTH
    min_consecutive_points: int = 3
    exclude_labels: Optional[list[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dataset_name: Optional[str] = None


class DetectAnomaliesMlArgs(BaseModel):
    power_column: Optional[str] = None
    wind_speed_column: Optional[str] = None
    setpoint_column: Optional[str] = None
    pitch_column: Optional[str] = None
    wind_speed_std_column: Optional[str] = None
    bin_width: float = TURBINE_DEFAULT_BIN_WIDTH
    dataset_name: Optional[str] = None


class PredictPowerArgs(BaseModel):
    power_column: Optional[str] = None
    wind_speed_column: Optional[str] = None
    setpoint_column: Optional[str] = None
    feature_columns: Optional[list[str]] = None
    bin_width: float = TURBINE_DEFAULT_BIN_WIDTH
    dataset_name: Optional[str] = None


class CalculateTurbineKpisArgs(BaseModel):
    power_column: Optional[str] = None
    wind_speed_column: Optional[str] = None
    setpoint_column: Optional[str] = None
    theoretical_power_column: Optional[str] = None
    bin_width: float = TURBINE_DEFAULT_BIN_WIDTH
    interval_minutes: Optional[float] = None
    dataset_name: Optional[str] = None


def execute_tool(tool_name: str, arguments: dict, dataset_manager: DatasetManager) -> dict:
    """
    Điểm vào DUY NHẤT để chạy 1 tool call từ LLM.
    Validate bằng Pydantic trước, bắt lỗi rõ ràng nếu argument sai type/thiếu field.
    Trả về luôn theo format {ok, data, error} — không bao giờ raise exception ra ngoài.
    """
    try:
        if tool_name == "get_dataset_summary":
            # Không có Pydantic model riêng (không tham số bắt buộc) — lấy dataset_name
            # trực tiếp từ dict, .get() tự trả None nếu thiếu key (đúng nghĩa optional).
            return get_dataset_summary(dataset_manager, dataset_name=arguments.get("dataset_name"))

        elif tool_name == "aggregate_metric":
            args = AggregateMetricArgs(**arguments)
            return aggregate_metric(
                dataset_manager,
                column=args.column,
                aggregation=args.aggregation,
                start_date=args.start_date,
                end_date=args.end_date,
                dataset_name=args.dataset_name,
            )

        elif tool_name == "group_metric":
            args = GroupMetricArgs(**arguments)
            return group_metric(
                dataset_manager,
                column=args.column,
                aggregation=args.aggregation,
                group_by=args.group_by,
                dataset_name=args.dataset_name,
            )

        elif tool_name == "find_peak":
            args = FindPeakArgs(**arguments)
            return find_peak(
                dataset_manager, column=args.column, mode=args.mode, dataset_name=args.dataset_name,
            )

        elif tool_name == "compare_periods":
            args = ComparePeriodsArgs(**arguments)
            # QUAN TRỌNG: signature thật ở tools/dataframe_tools.py (M2) đặt các
            # tham số period_* TRƯỚC aggregation — dùng keyword args để tránh
            # nhầm thứ tự vị trí.
            return compare_periods(
                dataset_manager,
                column=args.column,
                period_a_start=args.period_a_start,
                period_a_end=args.period_a_end,
                period_b_start=args.period_b_start,
                period_b_end=args.period_b_end,
                aggregation=args.aggregation,
                dataset_name=args.dataset_name,
            )

        elif tool_name == "calculate_correlation":
            args = CalculateCorrelationArgs(**arguments)
            return calculate_correlation(
                dataset_manager,
                column_a=args.column_a,
                column_b=args.column_b,
                start_date=args.start_date,
                end_date=args.end_date,
                dataset_name=args.dataset_name,
            )

        elif tool_name == "calculate":
            args = CalculateArgs(**arguments)
            return calculate(args.operation, args.value_a, args.value_b)

        elif tool_name == "search_knowledge_base":
            args = SearchKnowledgeBaseArgs(**arguments)
            return search_knowledge_base(
                args.query,
                top_k=args.top_k,
                language=args.language,
                document_type=args.document_type,
            )

        elif tool_name == "clean_missing_values":
            args = CleanMissingValuesArgs(**arguments)
            return clean_missing_values(
                dataset_manager, dataset_name=args.dataset_name, column=args.column, strategy=args.strategy,
            )

        elif tool_name == "detect_duplicates":
            args = DetectDuplicatesArgs(**arguments)
            return detect_duplicates(dataset_manager, dataset_name=args.dataset_name, subset_cols=args.subset_cols)

        elif tool_name == "detect_statistical_anomalies":
            args = DetectStatisticalAnomaliesArgs(**arguments)
            return detect_statistical_anomalies(
                dataset_manager, column=args.column, dataset_name=args.dataset_name,
                method=args.method, threshold=args.threshold,
            )

        elif tool_name == "correlation_analysis":
            args = CorrelationAnalysisArgs(**arguments)
            return correlation_analysis(
                dataset_manager, target_column=args.target_column, dataset_name=args.dataset_name, top_n=args.top_n,
            )

        elif tool_name == "explain_anomaly":
            args = ExplainAnomalyArgs(**arguments)
            return explain_anomaly(
                dataset_manager, dataset_name=args.dataset_name,
                anomaly_mask_column=args.anomaly_mask_column, row_indices=args.row_indices,
                compare_columns=args.compare_columns,
            )

        elif tool_name == "detect_physical_violations":
            args = DetectPhysicalViolationsArgs(**arguments)
            return detect_physical_violations(dataset_manager, dataset_name=args.dataset_name)

        elif tool_name == "detect_sensor_issues":
            args = DetectSensorIssuesArgs(**arguments)
            return detect_sensor_issues(
                dataset_manager, dataset_name=args.dataset_name,
                flatline_min_length=args.flatline_min_length, gap_multiplier=args.gap_multiplier,
            )

        elif tool_name == "diagnose_theoretical_power_deviation":
            args = DiagnoseTheoreticalPowerDeviationArgs(**arguments)
            return diagnose_theoretical_power_deviation(
                dataset_manager, dataset_name=args.dataset_name, threshold_pct=args.threshold_pct,
            )

        elif tool_name == "estimate_power_curve":
            args = EstimatePowerCurveArgs(**arguments)
            return estimate_power_curve_for_agent(
                dataset_manager, args.dataset_name, args.power_column, args.wind_speed_column, args.bin_width,
            )

        elif tool_name == "classify_turbine_status":
            args = ClassifyTurbineStatusArgs(**arguments)
            return classify_turbine_status_for_agent(
                dataset_manager, args.dataset_name, args.power_column, args.wind_speed_column,
                args.setpoint_column, args.bin_width,
            )

        elif tool_name == "group_anomaly_windows":
            args = GroupAnomalyWindowsArgs(**arguments)
            return group_anomaly_windows_for_agent(
                dataset_manager, args.dataset_name, args.timestamp_column, args.power_column,
                args.wind_speed_column, args.setpoint_column, args.bin_width,
                args.min_consecutive_points, args.exclude_labels,
                start_date=args.start_date, end_date=args.end_date,
            )

        elif tool_name == "detect_anomalies_ml":
            args = DetectAnomaliesMlArgs(**arguments)
            return detect_anomalies_ml_for_agent(
                dataset_manager, args.dataset_name, args.power_column, args.wind_speed_column,
                args.setpoint_column, args.pitch_column, args.wind_speed_std_column, args.bin_width,
            )

        elif tool_name == "predict_power":
            args = PredictPowerArgs(**arguments)
            return predict_power_for_agent(
                dataset_manager, args.dataset_name, args.power_column, args.wind_speed_column,
                args.setpoint_column, args.feature_columns, args.bin_width,
            )

        elif tool_name == "calculate_turbine_kpis":
            args = CalculateTurbineKpisArgs(**arguments)
            return calculate_turbine_kpis_for_agent(
                dataset_manager, args.dataset_name, args.power_column, args.wind_speed_column,
                args.setpoint_column, args.theoretical_power_column, args.bin_width, args.interval_minutes,
            )

        else:
            return failure(f"Unknown tool: '{tool_name}'")

    except ValidationError as e:
        return failure(f"Invalid arguments for tool '{tool_name}': {e}")
    except Exception as e:
        return failure(f"Tool '{tool_name}' execution failed: {e}")


TOOL_NAMES = [
    "get_dataset_summary", "aggregate_metric", "group_metric", "find_peak",
    "compare_periods", "calculate_correlation", "calculate", "search_knowledge_base",
    "clean_missing_values", "detect_duplicates", "detect_statistical_anomalies",
    "correlation_analysis", "explain_anomaly",
    "detect_physical_violations", "detect_sensor_issues", "diagnose_theoretical_power_deviation",
    "estimate_power_curve", "classify_turbine_status", "group_anomaly_windows",
    "detect_anomalies_ml", "predict_power", "calculate_turbine_kpis",
]
