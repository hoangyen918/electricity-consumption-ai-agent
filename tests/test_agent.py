"""
4 câu hỏi 'gate' bắt buộc pass trước khi coi agent hoàn thành cốt lõi.
Các test này gọi LLM THẬT qua OpenRouter — có cost, nhưng rất nhỏ (vài nghìn token/câu).
"""
import pytest
from pathlib import Path
from tools.dataset_manager import DatasetManager
from agent.agent import Agent


PROCESSED_CSV = "data/processed/tetouan_clean.csv"


@pytest.fixture
def agent():
    if not Path(PROCESSED_CSV).exists():
        pytest.fail(
            f"{PROCESSED_CSV} không tồn tại. Chạy: python scripts/prepare_data.py trước khi test."
        )
    manager = DatasetManager()
    manager.load_csv(PROCESSED_CSV)
    return Agent(manager)


def test_gate_1_simple_data_question(agent):
    """Gate 1: Mức tiêu thụ trung bình Zone 1 tháng 7 -> phải gọi aggregate_metric."""
    result = agent.ask("Mức tiêu thụ trung bình của Zone 1 trong tháng 7 là bao nhiêu?")

    assert not result["hit_max_iterations"]
    tool_names = [t["tool_name"] for t in result["trace"]]
    assert "aggregate_metric" in tool_names

    agg_call = next(t for t in result["trace"] if t["tool_name"] == "aggregate_metric")
    assert agg_call["result"]["ok"] is True
    assert agg_call["result"]["data"]["rows_used"] > 0

    print(f"\n[GATE 1] Answer: {result['answer']}")


def test_gate_2_multistep_data_question(agent):
    """Gate 2: Ngày Zone 1 đạt đỉnh, nhiệt độ TB ngày đó -> find_peak RỒI aggregate_metric."""
    result = agent.ask(
        "Vào ngày Zone 1 đạt mức tiêu thụ cao nhất, nhiệt độ trung bình ngày đó là bao nhiêu?"
    )

    assert not result["hit_max_iterations"]
    tool_names = [t["tool_name"] for t in result["trace"]]
    assert "find_peak" in tool_names
    assert "aggregate_metric" in tool_names
    # find_peak phải xảy ra TRƯỚC aggregate_metric (vì cần ngày từ find_peak)
    assert tool_names.index("find_peak") < tool_names.index("aggregate_metric")

    print(f"\n[GATE 2] Trace: {[t['tool_name'] for t in result['trace']]}")
    print(f"[GATE 2] Answer: {result['answer']}")


def test_gate_3_knowledge_question(agent):
    """Gate 3: Câu hỏi KB có ground truth đã verify -> search_knowledge_base + citation đúng."""
    result = agent.ask("Tỷ lệ hộ dân nông thôn có điện của EVN là bao nhiêu phần trăm?")

    assert not result["hit_max_iterations"]
    tool_names = [t["tool_name"] for t in result["trace"]]
    assert "search_knowledge_base" in tool_names

    # Ground truth đã verify thủ công (khớp với VERIFIED_CASES trong tests/test_retriever.py):
    # "Tỷ lệ hộ dân nông thôn có điện" -> EVN Annual Report 2024-2025, trang 9
    EXPECTED_SOURCE_FRAGMENT = "2024-2025"
    EXPECTED_PAGE = 9

    kb_call = next(t for t in result["trace"] if t["tool_name"] == "search_knowledge_base")
    assert kb_call["result"]["ok"] is True
    all_results = kb_call["result"]["data"]["results"]

    # Kiểm tra: đúng CẶP (nguồn, trang) phải nằm trong candidates trả về,
    # không chỉ riêng số trang -- vì 2 tài liệu khác nhau đều có "trang 9"
    # với nội dung hoàn toàn khác nhau (xem DATA_SOURCES.md - Known Limitations)
    correct_pair_in_candidates = any(
        EXPECTED_SOURCE_FRAGMENT in r["source"] and r["page"] == EXPECTED_PAGE
        for r in all_results
    )
    assert correct_pair_in_candidates, (
        f"Đúng nguồn ({EXPECTED_SOURCE_FRAGMENT}, trang {EXPECTED_PAGE}) không nằm trong "
        f"candidates trả về. Got: {[(r['source'], r['page']) for r in all_results]}"
    )

    # Answer phải chứa đúng số trang VÀ dấu hiệu đúng năm tài liệu (2024),
    # để phân biệt với trang 9 của tài liệu khác (2022-2023, nội dung sai)
    answer = result["answer"]
    assert str(EXPECTED_PAGE) in answer, f"Answer thiếu số trang {EXPECTED_PAGE}: {answer}"
    assert "2024" in answer, (
        f"Answer có số trang {EXPECTED_PAGE} nhưng không rõ trích dẫn đúng tài liệu 2024-2025 "
        f"(có thể đã nhầm sang trang 9 của tài liệu 2022-2023, vốn là nội dung chart bị vỡ): {answer}"
    )

    print(f"\n[GATE 3] Candidates: {[(r['source'], r['page']) for r in all_results]}")
    print(f"[GATE 3] Answer: {answer}")


def test_gate_4_both_data_and_kb(agent):
    """Gate 4: search_knowledge_base tự tra ra tỷ lệ % từ Chỉ thị 20, RỒI calculate áp dụng."""
    result = agent.ask(
        "Theo quy định, một cơ sở sản xuất tiêu thụ 1.2 triệu kWh điện mỗi năm "
        "cần tiết kiệm tối thiểu bao nhiêu kWh?"
    )

    assert not result["hit_max_iterations"]
    tool_names = [t["tool_name"] for t in result["trace"]]
    assert "search_knowledge_base" in tool_names
    assert "calculate" in tool_names
    # search_knowledge_base phải xảy ra TRƯỚC calculate (cần tỷ lệ % trước khi tính)
    assert tool_names.index("search_knowledge_base") < tool_names.index("calculate")

    kb_call = next(t for t in result["trace"] if t["tool_name"] == "search_knowledge_base")
    assert kb_call["result"]["ok"] is True

    calc_call = next(t for t in result["trace"] if t["tool_name"] == "calculate")
    assert calc_call["result"]["ok"] is True
    # Đúng theo Chỉ thị 20: 2% của 1,200,000 = 24,000 kWh
    assert calc_call["result"]["data"]["result"] == 24000

    # Answer phải chứa đúng con số kết quả VÀ có trích dẫn nguồn Chỉ thị 20
    assert "24" in result["answer"] and "000" in result["answer"], (
        f"Answer thiếu con số kết quả 24,000: {result['answer']}"
    )
    assert "Chỉ thị" in result["answer"] or "20/CT-TTg" in result["answer"], (
        f"Answer thiếu trích dẫn nguồn Chỉ thị 20: {result['answer']}"
    )

    print(f"\n[GATE 4] Trace: {tool_names}")
    print(f"[GATE 4] Answer: {result['answer']}")


def test_max_iterations_never_exceeded(agent):
    """Sanity check: iterations không bao giờ vượt MAX_AGENT_ITERATIONS."""
    from config.settings import SETTINGS
    result = agent.ask("Dataset có bao nhiêu dòng?")
    assert result["iterations"] <= SETTINGS.max_agent_iterations
