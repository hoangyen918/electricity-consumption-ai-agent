"""
Smoke test agent thật cho 13 tool turbine gió mới đăng ký (Phase 5) — gọi LLM THẬT qua
OpenRouter (cost nhỏ), theo đúng pattern tests/test_agent.py. Chỉ 3 test đại diện (không
lặp lại toàn bộ 8 câu đã verify thủ công khi phát triển — xem báo cáo Phase 5 cho log
đầy đủ 8 câu, bao gồm cả các bug đã phát hiện+sửa: naming dataset_name, hallucination
sau tool fail, thiếu dataset_name khi có nhiều dataset, group_anomaly_windows quá lớn).
"""
import pytest
from pathlib import Path

from tools.dataset_manager import DatasetManager
from agent.agent import Agent

GT1_PATH = "data/wind_turbine/data1/his_data_GT1.csv"
WT1_PATH = "data/wind_turbine/data2/WindFarm04_WT1.csv"


@pytest.fixture
def turbine_agent():
    if not Path(GT1_PATH).exists() or not Path(WT1_PATH).exists():
        pytest.skip("Dữ liệu turbine gió (data/wind_turbine/) chưa có trên máy này — bỏ qua (file không git-track).")
    manager = DatasetManager()
    manager.load_csv(GT1_PATH, name="his_data_GT1")
    manager.load_csv(WT1_PATH, name="WindFarm04_WT1")
    return Agent(manager)


def test_turbine_gate_1_physical_violations_routes_to_correct_tool_and_dataset(turbine_agent):
    """Câu hỏi rõ ràng về WT1 -> phải gọi detect_physical_violations VỚI dataset_name
    đúng (WindFarm04_WT1, không phải GT1 dù GT1 cũng đã tải)."""
    result = turbine_agent.ask("Có bất thường vật lý nào trong dữ liệu WT1 không?")

    assert not result["hit_max_iterations"]
    tool_names = [t["tool_name"] for t in result["trace"]]
    assert "detect_physical_violations" in tool_names

    call = next(t for t in result["trace"] if t["tool_name"] == "detect_physical_violations")
    assert call["result"]["ok"] is True
    assert call["arguments"].get("dataset_name") == "WindFarm04_WT1"
    assert call["result"]["data"]["site_detected"] == "site2"

    print(f"\n[TURBINE GATE 1] Answer: {result['answer']}")


def test_turbine_gate_2_classify_status_with_correct_setpoint_column(turbine_agent):
    """Câu hỏi phân loại trạng thái Site 1 -> classify_turbine_status với dataset_name=
    his_data_GT1 VÀ setpoint_column=GT1.SP.ActualSPP (để Curtailment nhận diện đúng)."""
    result = turbine_agent.ask("Phân loại trạng thái vận hành Site 1 theo 8 nhãn Meteodyn")

    assert not result["hit_max_iterations"]
    tool_names = [t["tool_name"] for t in result["trace"]]
    assert "classify_turbine_status" in tool_names

    call = next(t for t in result["trace"] if t["tool_name"] == "classify_turbine_status")
    assert call["result"]["ok"] is True
    assert call["arguments"].get("dataset_name") == "his_data_GT1"
    data = call["result"]["data"]
    assert "classification" not in data  # KHONG duoc lo dict day du hang tram nghin dong cho LLM
    assert "summary" in data
    # so lieu da validate that (xem docstring classify_turbine_status): Normal chiem da so
    assert data["summary"]["Normal"]["pct"] > 50

    print(f"\n[TURBINE GATE 2] Answer: {result['answer']}")


def test_turbine_gate_3_domain_term_question_uses_knowledge_base_not_turbine_tool(turbine_agent):
    """Câu hỏi định nghĩa thuật ngữ (không phải phân tích) -> phải dùng
    search_knowledge_base, KHÔNG được chọn nhầm 1 trong 13 tool turbine mới."""
    result = turbine_agent.ask("Curtailment nghĩa là gì?")

    assert not result["hit_max_iterations"]
    tool_names = [t["tool_name"] for t in result["trace"]]
    assert "search_knowledge_base" in tool_names

    turbine_tool_names = {
        "detect_physical_violations", "detect_sensor_issues", "diagnose_theoretical_power_deviation",
        "estimate_power_curve", "classify_turbine_status", "group_anomaly_windows",
        "detect_anomalies_ml", "predict_power",
    }
    assert not (set(tool_names) & turbine_tool_names), (
        f"Câu hỏi định nghĩa thuật ngữ KHÔNG nên gọi tool phân tích turbine, nhưng đã gọi: {set(tool_names) & turbine_tool_names}"
    )

    print(f"\n[TURBINE GATE 3] Answer: {result['answer']}")
