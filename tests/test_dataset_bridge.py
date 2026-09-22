"""
Test cho _bridge_wind_turbine_datasets_to_chat() / _ensure_wind_turbine_dataset_loaded()
sau khi bỏ gate từ khóa (TURBINE_QUESTION_KEYWORDS) và chuyển sang cơ chế
set_active=False + trigger ngay khi Dashboard load xong 1 dataset (xem docstring 2 hàm
này trong app.py).

Dùng AppTest.from_function() với harness NHỎ, dữ liệu GIẢ (DataFrame nhỏ qua
add_dataframe(), KHÔNG đọc file turbine thật) — chỉ test đúng cơ chế bridge/active,
KHÔNG cần dữ liệu turbine thật cho việc này (xem tests/test_app_ui.py cho lý do tương
tự: tránh chi phí tính toán 13 tab không liên quan tới điều đang test).
"""
from streamlit.testing.v1 import AppTest


def _bridge_harness():
    import pandas as pd
    import streamlit as st
    import app
    from tools.dataset_manager import DatasetManager

    if "dataset_manager" not in st.session_state:
        chat_manager = DatasetManager()
        chat_manager.add_dataframe(pd.DataFrame({"zone_1": [1, 2, 3]}), name="Tetouan")
        st.session_state.dataset_manager = chat_manager

    wt_manager = app._get_wind_turbine_manager()
    if "GT1_fake" not in wt_manager.list_datasets():
        wt_manager.add_dataframe(pd.DataFrame({"power": [10.0, 20.0, 30.0]}), name="GT1_fake")

    app._bridge_wind_turbine_datasets_to_chat()


def test_bridge_copies_turbine_dataset_into_chat_manager():
    at = AppTest.from_function(_bridge_harness, default_timeout=15)
    at.run()

    chat_manager = at.session_state["dataset_manager"]
    assert "GT1_fake" in chat_manager.list_datasets()


def test_bridge_does_not_change_chat_active_dataset():
    """Đây LÀ bug chính đã audit trước khi sửa: Dashboard load xong turbine dataset
    KHÔNG được phép cướp active của Chat — Tetouan (đã active TRƯỚC khi bridge chạy)
    phải vẫn là active SAU khi bridge chạy, dù dataset_manager giờ có 2 dataset."""
    at = AppTest.from_function(_bridge_harness, default_timeout=15)
    at.run()

    chat_manager = at.session_state["dataset_manager"]
    assert chat_manager.get_active_name() == "Tetouan"
    assert set(chat_manager.list_datasets()) == {"Tetouan", "GT1_fake"}


def _bridge_twice_harness():
    import pandas as pd
    import streamlit as st
    import app
    from tools.dataset_manager import DatasetManager

    if "dataset_manager" not in st.session_state:
        chat_manager = DatasetManager()
        chat_manager.add_dataframe(pd.DataFrame({"zone_1": [1, 2, 3]}), name="Tetouan")
        st.session_state.dataset_manager = chat_manager

    wt_manager = app._get_wind_turbine_manager()
    if "GT1_fake" not in wt_manager.list_datasets():
        wt_manager.add_dataframe(pd.DataFrame({"power": [10.0, 20.0, 30.0]}), name="GT1_fake")

    # Gọi 2 LẦN trong cùng 1 lượt — mô phỏng nhiều tab Dashboard cùng load xong dataset
    # turbine này và mỗi tab đều tự gọi _ensure_wind_turbine_dataset_loaded() (mỗi lần
    # đều kích hoạt bridge, xem app.py) — phải idempotent, KHÔNG lỗi, KHÔNG đổi active.
    app._bridge_wind_turbine_datasets_to_chat()
    app._bridge_wind_turbine_datasets_to_chat()


def test_bridge_is_idempotent_across_repeated_calls():
    at = AppTest.from_function(_bridge_twice_harness, default_timeout=15)
    at.run()
    assert not at.exception

    chat_manager = at.session_state["dataset_manager"]
    assert chat_manager.get_active_name() == "Tetouan"
    assert set(chat_manager.list_datasets()) == {"Tetouan", "GT1_fake"}
    assert "GT1_fake" in at.session_state["_bridged_turbine_names"]


def _bridge_skips_existing_name_harness():
    import pandas as pd
    import streamlit as st
    import app
    from tools.dataset_manager import DatasetManager

    if "dataset_manager" not in st.session_state:
        chat_manager = DatasetManager()
        # Chat đã có SẴN 1 dataset TRÙNG TÊN với dataset turbine sắp bridge (vd user tự
        # upload file cùng tên) — bridge KHÔNG được ghi đè dataset user đang có.
        chat_manager.add_dataframe(pd.DataFrame({"already": ["mine"]}), name="GT1_fake")
        st.session_state.dataset_manager = chat_manager

    wt_manager = app._get_wind_turbine_manager()
    if "GT1_fake" not in wt_manager.list_datasets():
        wt_manager.add_dataframe(pd.DataFrame({"power": [10.0, 20.0, 30.0]}), name="GT1_fake")

    app._bridge_wind_turbine_datasets_to_chat()


def test_bridge_does_not_overwrite_existing_dataset_with_same_name():
    at = AppTest.from_function(_bridge_skips_existing_name_harness, default_timeout=15)
    at.run()

    chat_manager = at.session_state["dataset_manager"]
    df = chat_manager.get_dataframe("GT1_fake")
    assert "already" in df.columns  # dataset GỐC của user còn nguyên, không bị ghi đè
