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
from rag.retriever import search_knowledge_base
from tools.tool_results import failure


AggregationType = Literal["mean", "sum", "min", "max", "median", "std", "count"]
GroupByType = Literal["hour", "day", "month", "weekday", "year", "date"]


class AggregateMetricArgs(BaseModel):
    column: str
    aggregation: AggregationType
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class GroupMetricArgs(BaseModel):
    column: str
    aggregation: AggregationType
    group_by: GroupByType


class FindPeakArgs(BaseModel):
    column: str
    mode: Literal["max", "min"] = "max"


class ComparePeriodsArgs(BaseModel):
    column: str
    period_a_start: str
    period_a_end: str
    period_b_start: str
    period_b_end: str
    aggregation: AggregationType = "mean"


class CalculateCorrelationArgs(BaseModel):
    column_a: str
    column_b: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CalculateArgs(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide", "percentage_of", "percentage_change"]
    value_a: float
    value_b: float


class SearchKnowledgeBaseArgs(BaseModel):
    query: str
    top_k: int = 8


def execute_tool(tool_name: str, arguments: dict, dataset_manager: DatasetManager) -> dict:
    """
    Điểm vào DUY NHẤT để chạy 1 tool call từ LLM.
    Validate bằng Pydantic trước, bắt lỗi rõ ràng nếu argument sai type/thiếu field.
    Trả về luôn theo format {ok, data, error} — không bao giờ raise exception ra ngoài.
    """
    try:
        if tool_name == "get_dataset_summary":
            return get_dataset_summary(dataset_manager)

        elif tool_name == "aggregate_metric":
            args = AggregateMetricArgs(**arguments)
            return aggregate_metric(
                dataset_manager,
                column=args.column,
                aggregation=args.aggregation,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        elif tool_name == "group_metric":
            args = GroupMetricArgs(**arguments)
            return group_metric(
                dataset_manager,
                column=args.column,
                aggregation=args.aggregation,
                group_by=args.group_by,
            )

        elif tool_name == "find_peak":
            args = FindPeakArgs(**arguments)
            return find_peak(dataset_manager, column=args.column, mode=args.mode)

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
            )

        elif tool_name == "calculate_correlation":
            args = CalculateCorrelationArgs(**arguments)
            return calculate_correlation(
                dataset_manager,
                column_a=args.column_a,
                column_b=args.column_b,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        elif tool_name == "calculate":
            args = CalculateArgs(**arguments)
            return calculate(args.operation, args.value_a, args.value_b)

        elif tool_name == "search_knowledge_base":
            args = SearchKnowledgeBaseArgs(**arguments)
            return search_knowledge_base(args.query, args.top_k)

        else:
            return failure(f"Unknown tool: '{tool_name}'")

    except ValidationError as e:
        return failure(f"Invalid arguments for tool '{tool_name}': {e}")
    except Exception as e:
        return failure(f"Tool '{tool_name}' execution failed: {e}")


TOOL_NAMES = [
    "get_dataset_summary", "aggregate_metric", "group_metric", "find_peak",
    "compare_periods", "calculate_correlation", "calculate", "search_knowledge_base",
]
