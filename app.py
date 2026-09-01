"""Streamlit UI cho Electricity Consumption AI Agent."""
import tempfile
from pathlib import Path

import streamlit as st

from tools.dataset_manager import DatasetManager
from agent.agent import Agent
from config.settings import SETTINGS
from rag.vector_store import count_documents

st.set_page_config(page_title=SETTINGS.app_title, page_icon="⚡", layout="wide")

DEFAULT_DATASET_PATH = "data/processed/tetouan_clean.csv"


def init_session_state():
    if "dataset_manager" not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()
        default_path = Path(DEFAULT_DATASET_PATH)
        if default_path.exists():
            st.session_state.dataset_manager.load_csv(str(default_path))
            st.session_state.dataset_loaded = True
            st.session_state.dataset_source = "Tetouan City Power Consumption (mặc định)"
        else:
            st.session_state.dataset_loaded = False
            st.session_state.dataset_source = None

    if "agent" not in st.session_state:
        st.session_state.agent = Agent(st.session_state.dataset_manager)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def handle_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        st.session_state.dataset_manager.load_file(tmp_path)
        st.session_state.dataset_loaded = True
        st.session_state.dataset_source = f"Upload: {uploaded_file.name}"
        st.session_state.agent.reset()
        st.session_state.chat_history = []
        st.sidebar.success(f"Đã tải {uploaded_file.name}")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Lỗi khi tải file: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def render_sidebar():
    with st.sidebar:
        st.header("⚡ Electricity Agent")
        st.caption(SETTINGS.app_title)

        st.subheader("📊 Dataset")
        if st.session_state.dataset_loaded:
            df = st.session_state.dataset_manager.get_dataframe()
            st.success(f"Đã tải: {st.session_state.dataset_source}")
            st.write(f"- Số dòng: {len(df):,}")
            st.write(f"- Số cột: {len(df.columns)}")
            if "datetime" in df.columns:
                st.write(f"- Khoảng thời gian: {df['datetime'].min()} → {df['datetime'].max()}")
        else:
            st.warning(
                "Chưa có dataset mặc định. Chạy `python scripts/prepare_data.py` "
                "hoặc upload file CSV/Excel bên dưới."
            )

        uploaded_file = st.file_uploader(
            "Upload dataset khác (CSV/Excel)", type=["csv", "xlsx", "xls"]
        )
        if uploaded_file is not None and st.session_state.get("_last_uploaded_file_id") != uploaded_file.file_id:
            st.session_state._last_uploaded_file_id = uploaded_file.file_id
            handle_upload(uploaded_file)

        st.divider()

        st.subheader("📚 Knowledge Base")
        try:
            kb_count = count_documents()
            st.success(f"{kb_count} đoạn tài liệu đã nạp")
        except Exception:
            st.error("Chưa kết nối được ChromaDB. Chạy `python scripts/ingest_kb.py` trước.")

        if "last_kb_ingest" in st.session_state:
            info = st.session_state["last_kb_ingest"]
            st.caption(
                f"Ingest gần nhất: {info['chunk_count']} chunks · {info['token_count']} tokens · "
                f"${info['cost_usd']:.6f}"
            )

        st.write("**Tài liệu đã nạp:**")
        try:
            from rag.kb_manager import list_sources_with_chunk_counts, delete_document, add_document
            sources = list_sources_with_chunk_counts()
            for src in sources:
                col_a, col_b = st.columns([4, 1])
                col_a.write(f"📄 {src['title']} ({src['chunk_count']} đoạn)")
                if col_b.button("🗑️", key=f"del_{src['id']}"):
                    deleted = delete_document(src["id"])
                    st.success(f"Đã xóa {deleted} đoạn từ '{src['title']}'")
                    st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi tải danh sách KB: {e}")

        st.write("**Thêm tài liệu mới vào KB:**")
        kb_file = st.file_uploader(
            "Upload PDF/TXT/MD", type=["pdf", "txt", "md"], key="kb_uploader"
        )
        kb_title = st.text_input("Tên hiển thị khi trích dẫn", key="kb_title_input")
        if kb_file is not None and kb_title:
            if st.button("📥 Nạp vào Knowledge Base"):
                try:
                    from rag.kb_manager import add_document
                    result = add_document(
                        file_bytes=kb_file.getvalue(),
                        original_filename=kb_file.name,
                        title=kb_title,
                    )
                    st.success(f"Đã nạp '{kb_title}': {result['chunk_count']} đoạn.")
                    st.session_state["last_kb_ingest"] = result
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi nạp tài liệu: {e}")

        st.divider()

        st.subheader("⚙️ Cấu hình")
        st.write(f"- Chat model: `{SETTINGS.chat_model}`")
        st.write(f"- Embedding model: `{SETTINGS.embedding_model}`")
        st.write(f"- Max iterations: {SETTINGS.max_agent_iterations}")

        st.divider()
        if st.button("🔄 Xóa lịch sử hội thoại"):
            st.session_state.agent.reset()
            st.session_state.chat_history = []
            st.rerun()


def render_trace_expander(trace_info: dict):
    with st.expander("🔍 Xem chi tiết xử lý (Agent Trace)"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Số bước lặp", trace_info["iterations"])
        col2.metric("Thời gian", f"{trace_info['latency_ms']:.0f} ms")
        col3.metric("Chi phí ước tính", f"${trace_info['estimated_cost_usd']:.6f}")

        if trace_info["trace"]:
            st.write("**Công cụ đã gọi:**")
            for t in trace_info["trace"]:
                status_icon = "✅" if t["result"]["ok"] else "❌"
                st.write(f"{status_icon} Bước {t['iteration']}: `{t['tool_name']}`")
                st.json(t["arguments"])
                if not t["result"]["ok"]:
                    st.error(t["result"]["error"])
        else:
            st.write("_Không có công cụ nào được gọi (trả lời trực tiếp từ model)._")


def render_chat():
    st.title("⚡ Electricity Consumption AI Agent")
    st.caption("Hỏi bằng tiếng Việt về dữ liệu tiêu thụ điện hoặc tra cứu tài liệu EVN.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "trace_info" in msg:
                render_trace_expander(msg["trace_info"])

    question = st.chat_input("Nhập câu hỏi của bạn...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            if not st.session_state.dataset_loaded:
                answer = "Chưa có dataset nào được tải. Vui lòng upload file CSV/Excel ở sidebar trước."
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Đang xử lý..."):
                    try:
                        result = st.session_state.agent.ask(question)
                        st.write(result["answer"])
                        trace_info = {
                            "iterations": result["iterations"],
                            "latency_ms": result["latency_ms"],
                            "estimated_cost_usd": result["estimated_cost_usd"],
                            "trace": result["trace"],
                        }
                        render_trace_expander(trace_info)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "trace_info": trace_info,
                        })
                    except Exception as e:
                        error_msg = f"Đã xảy ra lỗi khi xử lý câu hỏi: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": error_msg}
                        )


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
