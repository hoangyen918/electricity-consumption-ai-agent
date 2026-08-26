import pytest
from tools.dataset_manager import DatasetManager
from agent.tool_registry import execute_tool


@pytest.fixture
def manager():
    m = DatasetManager()
    m.load_csv("data/sample/tetouan_sample.csv")
    return m


def test_get_dataset_summary(manager):
    result = execute_tool("get_dataset_summary", {}, manager)
    assert result["ok"] is True


def test_aggregate_metric_valid(manager):
    result = execute_tool(
        "aggregate_metric",
        {"column": "zone_1_consumption", "aggregation": "mean"},
        manager,
    )
    assert result["ok"] is True


def test_aggregate_metric_invalid_column(manager):
    result = execute_tool(
        "aggregate_metric",
        {"column": "khong_ton_tai", "aggregation": "mean"},
        manager,
    )
    assert result["ok"] is False


def test_aggregate_metric_invalid_aggregation_rejected_by_pydantic(manager):
    """Pydantic phải reject TRƯỚC KHI chạm Pandas — test literal enum validation."""
    result = execute_tool(
        "aggregate_metric",
        {"column": "zone_1_consumption", "aggregation": "run_python_code"},
        manager,
    )
    assert result["ok"] is False
    assert "invalid arguments" in result["error"].lower()


def test_calculate_percentage_of(manager):
    result = execute_tool(
        "calculate",
        {"operation": "percentage_of", "value_a": 1200000, "value_b": 2},
        manager,
    )
    assert result["ok"] is True
    assert result["data"]["result"] == 24000


def test_calculate_missing_required_field_rejected(manager):
    """Thiếu value_b — Pydantic phải bắt lỗi."""
    result = execute_tool(
        "calculate",
        {"operation": "add", "value_a": 5},
        manager,
    )
    assert result["ok"] is False


def test_unknown_tool_name(manager):
    result = execute_tool("delete_all_data", {}, manager)
    assert result["ok"] is False
    assert "unknown tool" in result["error"].lower()


def test_search_knowledge_base_via_registry(manager):
    """Test tool RAG được gọi đúng qua registry — gọi API thật, cost rất nhỏ."""
    result = execute_tool(
        "search_knowledge_base",
        {"query": "Tỷ lệ tổn thất điện năng là bao nhiêu?"},
        manager,
    )
    assert result["ok"] is True
    assert len(result["data"]["results"]) > 0
