"""
Test cho ĐÚNG kịch bản bug đã audit ("Hợp nhất wind_turbine_manager/dataset_manager",
2026-09): mở Dashboard (nạp + bridge 1 dataset turbine vào dataset_manager qua
set_active=False), sau đó Chat hỏi 1 câu về Tetouan KHÔNG nêu tên dataset — phải tính
trên Tetouan, KHÔNG được âm thầm lệch sang turbine.

KHÔNG gọi LLM thật (API key OpenRouter đang hết hạn — xác nhận lỗi 401 lúc viết test
này, xem báo cáo audit). Mock CHỈ agent.loop.chat_completion (kịch bản tool_call cố
định), để agent/tool_registry.py::execute_tool() và tools/dataframe_tools.py chạy THẬT
— đây chính là phần code cần verify (guard + active-dataset), không phải hành vi LLM.

Bằng chứng KHÔNG gián tiếp: spy trực tiếp lên DatasetManager.get_dataframe() để bắt
CHÍNH XÁC object DataFrame mà mỗi lần gọi trả về, so sánh identity (`is`) với DataFrame
Tetouan gốc đã load — không suy luận qua active_name/list_datasets() "trông có vẻ đúng".
"""
from streamlit.testing.v1 import AppTest

# Lưu ý: mọi import (pandas, MagicMock, patch...) đều khai báo LẠI bên TRONG
# _dashboard_then_chat_harness(), không ở module level — AppTest.from_function() chỉ
# trích xuất source CỦA RIÊNG hàm harness, không kế thừa import ở module cha (xem
# tests/test_dataset_bridge.py cho cùng lưu ý).


def _dashboard_then_chat_harness():
    import pandas as pd
    import streamlit as st
    from unittest.mock import MagicMock, patch
    import app
    import tools.dataset_manager as dsm_module
    from agent.agent import Agent

    # Path tương đối theo cwd (pytest chạy từ project root) — KHÔNG dùng __file__:
    # AppTest.from_function() trích xuất source CỦA RIÊNG hàm này ra 1 file tạm rồi
    # chạy, nên __file__ bên trong sẽ trỏ vào file tạm đó, không phải file test thật.
    sample_path = "data/sample/tetouan_sample.csv"

    # --- Setup: Chat đã có Tetouan active (mô phỏng init_session_state() thật) ---
    if "dataset_manager" not in st.session_state:
        chat_manager = dsm_module.DatasetManager()
        chat_manager.load_csv(sample_path, name="Tetouan")
        st.session_state.dataset_manager = chat_manager
        st.session_state.agent = Agent(chat_manager)
        st.session_state.chat_history = []
        st.session_state.dataset_loaded = True
        st.session_state.session_total_cost_usd = 0.0

    tetouan_df_ref = st.session_state.dataset_manager.get_dataframe("Tetouan")

    # --- Mô phỏng "mở Dashboard": 1 tab nạp xong 1 dataset turbine (dữ liệu giả nhỏ,
    # cột KHÁC HẲN Tetouan để nếu lỡ tool chạy nhầm dataset sẽ lộ ra ngay qua lỗi
    # "column not found" thay vì âm thầm trả số sai) -> trigger bridge NGAY (đúng thiết
    # kế mới, không đợi Chat hỏi gì) ---
    wt_manager = app._get_wind_turbine_manager()
    if "GT1_fake" not in wt_manager.list_datasets():
        wt_manager.add_dataframe(pd.DataFrame({"turbine_power_kw": [100.0, 200.0, 300.0]}), name="GT1_fake")
    app._bridge_wind_turbine_datasets_to_chat()

    # Kiểm tra TRỰC TIẾP fix set_active=False (KHÔNG bị guard mục 3 che mất — guard chỉ
    # chặn khi dataset_name=None, còn active_name có bị cướp hay không là chuyện KHÁC
    # hoàn toàn, phải tự verify riêng, không suy luận qua kết quả tool call phía dưới).
    # Xác nhận thật: đã thử tái tạo bug cũ (set_active=True) và assert này FAIL đúng
    # như kỳ vọng trước khi viết bản fix (2026-09-22).
    st.session_state["_active_name_after_bridge"] = st.session_state.dataset_manager.get_active_name()

    # --- Spy get_dataframe() để bắt CHÍNH XÁC object trả về mỗi lần gọi (bằng chứng
    # trực tiếp, không suy luận qua active_name) ---
    original_get_dataframe = dsm_module.DatasetManager.get_dataframe
    captured_calls = []

    def _spy_get_dataframe(self, name=None):
        result = original_get_dataframe(self, name)
        captured_calls.append((name, result))
        return result

    # --- Kịch bản LLM giả lập (KHÔNG gọi API thật):
    # Turn 1: gọi aggregate_metric KHÔNG nêu dataset_name (đúng hành vi "LLM đôi khi bỏ
    #   qua hướng dẫn" mà guard mới (_require_dataset_name_if_ambiguous) được viết ra để
    #   chặn) -> guard phải trả ok=False, KHÔNG được âm thầm chạy trên dataset nào.
    # Turn 2: LLM (giả lập) đọc lỗi, retry với dataset_name="Tetouan" tường minh -> phải
    #   thành công, dùng ĐÚNG DataFrame Tetouan.
    # Turn 3: trả lời cuối, không tool_call nào nữa.
    def _msg(tool_calls, content, finish_reason):
        m = MagicMock()
        m.content = content
        m.tool_calls = tool_calls
        m.model_dump.return_value = {
            "role": "assistant", "content": content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (tool_calls or [])
            ] or None,
        }
        return m

    def _resp(message, finish_reason):
        r = MagicMock()
        r.choices = [MagicMock(message=message, finish_reason=finish_reason)]
        usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        usage.model_dump.return_value = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        r.usage = usage
        return r

    tc1 = MagicMock(id="call_1")
    tc1.function.name = "aggregate_metric"
    tc1.function.arguments = '{"column": "zone_1_consumption", "aggregation": "mean"}'

    tc2 = MagicMock(id="call_2")
    tc2.function.name = "aggregate_metric"
    tc2.function.arguments = '{"column": "zone_1_consumption", "aggregation": "mean", "dataset_name": "Tetouan"}'

    responses = [
        _resp(_msg([tc1], None, "tool_calls"), "tool_calls"),
        _resp(_msg([tc2], None, "tool_calls"), "tool_calls"),
        _resp(_msg(None, "Mức tiêu thụ trung bình Zone 1 là X kWh.", "stop"), "stop"),
    ]

    with patch.object(dsm_module.DatasetManager, "get_dataframe", _spy_get_dataframe), \
         patch("agent.loop.chat_completion", side_effect=responses):
        app._handle_chat_question("Zone 1 tiêu thụ trung bình bao nhiêu?")

    st.session_state["_captured_calls"] = captured_calls
    st.session_state["_tetouan_df_ref"] = tetouan_df_ref
    st.session_state["_final_trace"] = st.session_state.chat_history[-1]["trace_info"]["trace"]


def test_bridge_does_not_hijack_chat_active_dataset():
    """Kiểm tra TRỰC TIẾP fix set_active=False — TÁCH RIÊNG khỏi guard mục 3 (guard chỉ
    ảnh hưởng lúc gọi tool với dataset_name=None, không nói lên được active_name có bị
    đổi hay không). Đã tự tái tạo bug cũ (đổi tạm set_active=True trong app.py) và xác
    nhận assertion này FAIL đúng như kỳ vọng trước khi có bản fix — assertion này thật
    sự "có răng", không phải test cho có."""
    at = AppTest.from_function(_dashboard_then_chat_harness, default_timeout=15)
    at.run()
    assert not at.exception

    assert at.session_state["_active_name_after_bridge"] == "Tetouan"


def test_ambiguous_first_call_rejected_by_guard_not_silently_wrong():
    """Turn 1 (aggregate_metric KHÔNG nêu dataset_name, giữa lúc có 2 dataset) PHẢI bị
    guard chặn (ok=False) — đây là bằng chứng trực tiếp guard hoạt động thật trong toàn
    bộ vòng lặp agent thật (không phải unit test riêng lẻ resolve_power_wind_columns/
    _require_dataset_name_if_ambiguous mà là qua đúng luồng execute_tool thật)."""
    at = AppTest.from_function(_dashboard_then_chat_harness, default_timeout=15)
    at.run()
    assert not at.exception

    trace = at.session_state["_final_trace"]
    assert len(trace) == 2
    assert trace[0]["tool_name"] == "aggregate_metric"
    assert trace[0]["result"]["ok"] is False
    assert "PHẢI chỉ định rõ dataset_name" in trace[0]["result"]["error"]


def test_retry_with_explicit_dataset_name_uses_real_tetouan_dataframe():
    """Turn 2 (retry với dataset_name='Tetouan' tường minh) PHẢI thành công VÀ DataFrame
    mà get_dataframe() trả về (bắt qua spy, KHÔNG suy luận gián tiếp) phải CHÍNH XÁC
    (identity, `is`) là DataFrame Tetouan gốc đã load lúc setup — không phải bản copy,
    không phải dataset turbine."""
    at = AppTest.from_function(_dashboard_then_chat_harness, default_timeout=15)
    at.run()
    assert not at.exception

    trace = at.session_state["_final_trace"]
    assert trace[1]["result"]["ok"] is True

    captured = at.session_state["_captured_calls"]
    tetouan_df = at.session_state["_tetouan_df_ref"]

    explicit_tetouan_calls = [df for name, df in captured if name == "Tetouan"]
    assert len(explicit_tetouan_calls) >= 1
    for df in explicit_tetouan_calls:
        assert df is tetouan_df, "get_dataframe('Tetouan') phải trả ĐÚNG object Tetouan gốc, không phải bản copy/dataset khác"


def test_no_get_dataframe_call_ever_silently_resolved_to_turbine():
    """Xuyên suốt toàn bộ luồng (cả 2 turn tool_call), KHÔNG lần get_dataframe() nào —
    dù gọi với name=None hay tên cụ thể — được phép trả về DataFrame turbine (cột
    "turbine_power_kw", hoàn toàn khác Tetouan). Đây là bằng chứng phủ định trực tiếp:
    bug hijack active-dataset (audit trước khi sửa) sẽ làm assertion này FAIL nếu tái
    diễn, vì lúc đó get_dataframe(None) sẽ trả turbine thay vì Tetouan."""
    at = AppTest.from_function(_dashboard_then_chat_harness, default_timeout=15)
    at.run()
    assert not at.exception

    captured = at.session_state["_captured_calls"]
    for name, df in captured:
        if df is not None and "turbine_power_kw" in df.columns:
            assert name == "GT1_fake", (
                f"get_dataframe(name={name!r}) trả về DataFrame turbine ngoài ý muốn — "
                "đúng bug hijack active-dataset đã audit."
            )
