"""File này GỌI API THẬT (cùng quy ước với tests/test_retriever.py) — không mock gì, xem
chi tiết dưới. Project CHƯA có cơ chế pytest marker nào (không có pytest.ini/conftest.py/
pyproject.toml, đã kiểm tra 2026-09-24) nên không gắn @pytest.mark riêng — nếu cần loại
trừ khỏi 1 lần chạy full suite, chạy chọn lọc bằng path (`pytest tests/ --ignore=tests/test_chat_kb_smoke.py`)
thay vì marker.

Smoke test: 1 câu hỏi KB đi qua ĐÚNG đường Chat UI thật (app._handle_chat_question ->
Agent.ask()), KHÔNG mock chat_completion/embedding — khác test_dashboard_chat_consistency.py
(mock chat_completion vì lúc viết OPENROUTER_API_KEY đã hết hạn). Verify toàn bộ pipeline
thật (embedding self-host TEI + chat model qua CHAT_BASE_URL) sau đợt đổi embedding
2026-09-22/24, xem DATA_SOURCES.md.

CHI PHÍ: 1 lần gọi embedding thật (self-host, miễn phí — xem services/cost_tracker.py)
+ 1 lần gọi chat model thật qua CHAT_BASE_URL (nhỏ, cùng cấp độ chi phí với các lần gọi
Agent.ask() thủ công đã chấp nhận nhiều lần trong buổi 2026-09-24) — CHỈ 1 câu hỏi, không
lặp lại nhiều lần/nhiều case.
"""
from streamlit.testing.v1 import AppTest


def _kb_chat_smoke_harness():
    import streamlit as st
    import app
    from tools.dataset_manager import DatasetManager
    from agent.agent import Agent

    sample_path = "data/sample/tetouan_sample.csv"
    manager = DatasetManager()
    manager.load_csv(sample_path, name="Tetouan")
    st.session_state.dataset_manager = manager
    st.session_state.agent = Agent(manager)
    st.session_state.chat_history = []
    st.session_state.dataset_loaded = True
    st.session_state.session_total_cost_usd = 0.0

    # Câu hỏi + đáp án đã biết trước (case kb_002 trong evals/cases.jsonl) — vừa verify
    # PASS qua eval kb_qa thật 2026-09-24 (rank=1, citation_correct=true, xem
    # evals/results/eval_detail_20260924_105458.json và DATA_SOURCES.md), và qua
    # Agent.ask() thủ công cùng ngày: trả lời "253,5 tỷ kWh", cite "EVN Annual Report
    # 2022-2023, p.10". Dùng lại đúng câu này để so khớp với kết quả đã biết, không
    # chọn câu mới chưa có ground truth xác nhận.
    app._handle_chat_question("Tổng sản lượng điện thương phẩm năm 2023 là bao nhiêu?")


def test_kb_question_gets_real_answer_with_correct_citation_through_chat_ui():
    """Gọi ĐÚNG đường thật app._handle_chat_question() -> Agent.ask() -> chat model
    thật + search_knowledge_base thật (embedding self-host TEI) -- KHÔNG mock gì. Xác
    nhận: không exception, có câu trả lời thật, citation đúng nguồn kỳ vọng."""
    at = AppTest.from_function(_kb_chat_smoke_harness, default_timeout=90)
    at.run()
    assert not at.exception, f"Lỗi khi hỏi qua Chat UI thật: {at.exception}"

    chat_history = at.session_state["chat_history"]
    assert len(chat_history) == 2, f"Kỳ vọng đúng 2 tin (user + assistant), có: {len(chat_history)}"
    assert chat_history[0]["role"] == "user"
    assistant_msg = chat_history[1]
    assert assistant_msg["role"] == "assistant"

    answer = assistant_msg["content"]
    assert answer, "Câu trả lời rỗng"
    assert "2022-2023" in answer, (
        f"Câu trả lời không cite đúng nguồn kỳ vọng (EVN Annual Report 2022-2023): {answer!r}"
    )
