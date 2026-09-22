"""
AppTest cho các thay đổi UI thuần Streamlit trong app.py (KHÔNG đụng logic tính
toán/agent loop/luồng dữ liệu — xem yêu cầu "Bố cục lại 13 tab Dashboard", 2026-09).

Dùng AppTest.from_function() với 1 harness NHỎ (chỉ gọi đúng hàm UI cần test) thay vì
AppTest.from_file("app.py") chạy toàn bộ Dashboard — 13 tab tính KPI thật trên các file
turbine gió lớn (WT1-4 ~29MB/file) MỖI LẦN rerun (st.tabs() chạy lại toàn bộ thân mọi
tab, không chỉ tab đang chọn — xem comment trong app.py::_render_power_curve_tab()),
từng gây timeout >180s khi test qua AppTest đầy đủ. Harness ở đây KHÔNG init dataset
manager/agent thật, chỉ import app.py (side-effect duy nhất: st.set_page_config, các
import module — không load dataset/gọi API) rồi gọi thẳng hàm UI cần kiểm tra.
"""
import pytest
from streamlit.testing.v1 import AppTest

APP_PY_PATH = "../app.py"  # AppTest.from_file resolves relative paths vs file này


def _side_chat_drawer_harness():
    import streamlit as st
    import app
    st.session_state.setdefault("chat_history", [])
    app._render_side_chat_drawer()


def test_old_floating_dialog_modal_removed():
    """Xác nhận modal @st.dialog cũ (_render_floating_chat_dialog/_close_floating_chat)
    đã bị xóa hoàn toàn, thay bằng _render_side_chat_drawer — không còn cách nào trong
    app.py mở lại modal giữa màn hình cũ."""
    import app
    assert not hasattr(app, "_render_floating_chat_dialog")
    assert not hasattr(app, "_close_floating_chat")
    assert hasattr(app, "_render_side_chat_drawer")


def test_side_chat_drawer_closed_by_default_no_chat_input():
    """Mặc định drawer ĐÓNG — chỉ có 1 nút tab mỏng (side_chat_tab_btn), KHÔNG render
    chat_input/nội dung lịch sử bên trong (tránh chiếm chỗ/tính toán thừa khi đóng)."""
    at = AppTest.from_function(_side_chat_drawer_harness, default_timeout=15)
    at.run()

    assert len(at.button) == 1
    assert at.button[0].key == "side_chat_tab_btn"
    assert len(at.chat_input) == 0


def test_side_chat_drawer_opens_on_tab_click():
    """Bấm tab mở drawer — chat_input (key=side_chat_input) xuất hiện, cờ
    show_side_chat=True được set trong session_state."""
    at = AppTest.from_function(_side_chat_drawer_harness, default_timeout=15)
    at.run()

    at.button(key="side_chat_tab_btn").click().run()

    assert at.session_state["show_side_chat"] is True
    assert len(at.chat_input) == 1
    assert at.chat_input[0].key == "side_chat_input"


def test_side_chat_drawer_closes_on_second_click():
    """Bấm lại đúng nút đó lần 2 (giờ hiển thị icon ✕) để đóng — chat_input biến mất."""
    at = AppTest.from_function(_side_chat_drawer_harness, default_timeout=15)
    at.run()
    at.button(key="side_chat_tab_btn").click().run()
    assert at.session_state["show_side_chat"] is True

    at.button(key="side_chat_tab_btn").click().run()

    assert at.session_state["show_side_chat"] is False
    assert len(at.chat_input) == 0


def test_side_chat_drawer_fragment_toggle_no_crash_on_real_app():
    """CHẬM (~1 phút, tính KPI thật trên toàn bộ Dashboard 13 tab lần đầu) — nhưng CẦN
    THIẾT: đây là test hồi quy cho 1 bug crash thật đã tìm thấy (2026-09-22) mà harness
    nhẹ ở các test phía trên KHÔNG bắt được (chỉ lộ ra khi chạy qua AppTest.from_file
    trên TOÀN BỘ app.py thật, không phải hàm _render_side_chat_drawer() gọi riêng lẻ):
    st.rerun(scope="fragment") ném StreamlitAPIException khi gọi ngay trong lượt xử lý
    click ĐẦU TIÊN của 1 widget mới trong @st.fragment (xem docstring
    _render_side_chat_drawer() — đã đọc source code Streamlit thật để xác nhận, không
    đoán). Test này click MỞ rồi ĐÓNG liên tiếp trên Dashboard thật, assert KHÔNG có
    exception nào (at.exception rỗng sau mỗi click) — đây là điều kiện tối thiểu để
    không tái diễn bug crash đó, KHÔNG assert thời gian chạy (dễ flaky giữa các máy) dù
    đã đo thật lúc viết test: click #1 (mở, lần đầu tương tác fragment) ~11.7s (vẫn
    tương đương tính lại cả 13 tab — đặc điểm ĐÃ BIẾT của Streamlit cho lần tương tác
    ĐẦU TIÊN với 1 fragment mới), click #2 (đóng) ~2.3s (~5 lần nhanh hơn — xác nhận cô
    lập rerun có tác dụng thật từ lần tương tác thứ 2 trở đi)."""
    at = AppTest.from_file(APP_PY_PATH, default_timeout=300)
    at.session_state["view_mode"] = "📊 Wind Turbine Dashboard"
    at.run()
    assert not at.exception, f"Lỗi ngay lần render Dashboard đầu tiên: {at.exception}"

    at.button(key="side_chat_tab_btn").click().run()
    assert not at.exception, f"Lỗi khi bấm MỞ drawer: {at.exception}"
    assert at.session_state["show_side_chat"] is True
    assert len(at.chat_input) == 1 and at.chat_input[0].key == "side_chat_input"

    at.button(key="side_chat_tab_btn").click().run()
    assert not at.exception, f"Lỗi khi bấm ĐÓNG drawer: {at.exception}"
    assert at.session_state["show_side_chat"] is False
    assert len(at.chat_input) == 0
