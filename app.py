"""Streamlit UI cho Electricity Consumption AI Agent."""
import json
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from tools.dataset_manager import DatasetManager
from agent.agent import Agent
from config.settings import SETTINGS
from rag.vector_store import count_documents
from rag.retriever import SCORE_THRESHOLD
from evals.run_evals import REFUSAL_PATTERNS, THRESHOLDS

st.set_page_config(page_title=SETTINGS.app_title, page_icon="⚡", layout="wide")

DEFAULT_DATASET_PATH = "data/processed/tetouan_clean.csv"
BASELINE_PATH = Path("evals/baselines/v1_baseline.json")
CASES_PATH = Path("evals/cases.jsonl")
MARGIN_WARNING_THRESHOLD = 0.1  # cảnh báo nếu distance cách SCORE_THRESHOLD dưới 0.1


def load_eval_baseline() -> dict | None:
    """Đọc evals/baselines/v1_baseline.json. Trả None nếu chưa tồn tại (vd clone mới
    chưa chạy eval) — KHÔNG raise, để UI tự xử lý hiển thị thông báo phù hợp."""
    if not BASELINE_PATH.exists():
        return None
    try:
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_case_questions() -> dict[str, str]:
    """Đọc evals/cases.jsonl -> {case_id: question}. Trả {} nếu file không tồn tại."""
    if not CASES_PATH.exists():
        return {}
    questions = {}
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            case = json.loads(line)
            questions[case["id"]] = case["question"]
    return questions


def init_session_state():
    if "dataset_manager" not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()
        default_path = Path(DEFAULT_DATASET_PATH)
        if default_path.exists():
            st.session_state.dataset_manager.load_csv(
                str(default_path), name="Tetouan City Power Consumption (mặc định)"
            )
            st.session_state.dataset_loaded = True
        else:
            st.session_state.dataset_loaded = False

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
        st.session_state.dataset_manager.load_file(tmp_path, name=uploaded_file.name)
        st.session_state.dataset_loaded = True
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

        total_cost = sum(
            msg["trace_info"]["estimated_cost_usd"]
            for msg in st.session_state.get("chat_history", [])
            if "trace_info" in msg
        )
        if total_cost > 0:
            st.metric("💰 Tổng chi phí câu hỏi (phiên này)", f"${total_cost:.6f}")

        st.subheader("📊 Dataset")
        manager = st.session_state.dataset_manager
        if st.session_state.dataset_loaded:
            active_name = manager.get_active_name()
            st.write("**Dataset đã tải:**")
            for name in manager.list_datasets():
                df_name = manager.get_dataframe(name)
                is_active = name == active_name
                label = (
                    f"{'⭐ ' if is_active else ''}{name} "
                    f"({len(df_name):,} dòng, {len(df_name.columns)} cột)"
                )
                col_a, col_b = st.columns([4, 1])
                col_a.write(label)
                if col_b.button("🗑️", key=f"del_dataset_{name}"):
                    manager.remove_dataset(name)
                    st.session_state.dataset_loaded = manager.is_loaded()
                    st.session_state.agent.reset()
                    st.session_state.chat_history = []
                    st.rerun()

            dataset_names = manager.list_datasets()
            if len(dataset_names) > 1:
                selected = st.sidebar.radio(
                    "Xem trước dataset:",
                    dataset_names,
                    index=dataset_names.index(manager.get_active_name()),
                )
                if selected != manager.get_active_name():
                    manager.switch_dataset(selected)
                    st.rerun()

            if active_name:
                active_df = manager.get_dataframe(active_name)
                st.caption(f"Dataset active (⭐): **{active_name}**")
                if "datetime" in active_df.columns:
                    st.write(
                        f"- Khoảng thời gian: {active_df['datetime'].min()} → "
                        f"{active_df['datetime'].max()}"
                    )
                st.write("**Preview:**")
                st.dataframe(active_df.head(), width="stretch")
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
                icon = "⚠️ 📄" if src.get("is_orphan") else "📄"
                label = f"{icon} {src['title']} ({src['chunk_count']} đoạn)"
                if src.get("is_orphan"):
                    label += " — thiếu metadata, nên xóa"
                col_a.write(label)
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

        st.sidebar.divider()
        if st.sidebar.button("📊 Xem đánh giá hệ thống"):
            eval_dashboard_dialog()


def render_eval_dashboard():
    """Phần 1 — trang tổng quan eval batch (53 case), đọc từ evals/baselines/v1_baseline.json
    (file thật do evals/run_evals.py sinh ra). Không hardcode số liệu."""
    baseline = load_eval_baseline()
    if baseline is None:
        st.warning("Chưa có dữ liệu đánh giá — chạy `python evals/run_evals.py` trước.")
        return

    st.caption(f"Baseline đo lúc: {baseline.get('timestamp', 'không rõ')}")

    # --- 1a. Bảng tổng quan ---
    st.subheader("📈 Tổng quan")
    total = baseline["total_cases"]
    passed = baseline["passed"]
    accuracy = baseline["overall_accuracy"]
    retrieval = baseline.get("retrieval", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall accuracy", f"{passed}/{total}", f"{accuracy*100:.1f}%")
    col2.metric("Recall@5", f"{retrieval.get('recall_at_5', 0)*100:.1f}%")
    col3.metric("Recall@10", f"{retrieval.get('recall_at_10', 0)*100:.1f}%")
    col4.metric("MRR", f"{retrieval.get('mrr', 0):.4f}")
    col5.metric("Citation correctness", f"{retrieval.get('citation_correctness_rate', 0)*100:.1f}%")

    st.write("**Threshold PASS/FAIL:**")
    threshold_values = {
        "overall_accuracy_min": accuracy,
        "recall_at_10_min": retrieval.get("recall_at_10", 0),
        "mrr_min": retrieval.get("mrr", 0),
    }
    for name, minimum in THRESHOLDS.items():
        actual = threshold_values.get(name)
        if actual is None:
            continue
        if actual >= minimum:
            st.success(f"✅ PASS — {name}: {actual:.4f} (ngưỡng ≥ {minimum})")
        else:
            st.error(f"❌ FAIL — {name}: {actual:.4f} (ngưỡng ≥ {minimum})")

    notes = baseline.get("notes")
    if notes:
        with st.expander("Ghi chú vận hành (network issue, giới hạn đã biết...)"):
            st.json(notes)

    st.divider()

    # --- 1b. Bảng per-category ---
    st.subheader("📂 Theo category")
    categories = baseline.get("categories") or baseline.get("by_category") or {}
    for cat, v in categories.items():
        cat_accuracy = v["accuracy"]
        line = f"**{cat}**: {v['passed']}/{v['total']} ({cat_accuracy*100:.1f}%)"
        if cat_accuracy >= 1.0:
            st.success(line)
        else:
            st.warning(line)

    st.divider()

    # --- 1c. Bảng per-case retrieval ---
    st.subheader("🔍 Chi tiết retrieval theo câu (kb_qa + both)")
    per_case = retrieval.get("per_case") or []
    if not per_case:
        st.write("_Không có dữ liệu per-case retrieval trong baseline này._")
    else:
        questions = load_case_questions()
        rows = []
        for c in per_case:
            rows.append({
                "id": c["id"],
                "câu hỏi": questions.get(c["id"], ""),
                "recall@5": c["recall_5"],
                "recall@10": c["recall_10"],
                # rank giữ nguyên kiểu số (None -> NaN qua pandas) — KHÔNG trộn str/int
                # trong cùng cột, vì Arrow serialization (st.dataframe) crash với cột
                # kiểu hỗn hợp. NaN tự hiển thị rõ trên UI, không cần chuỗi thay thế.
                "rank": c["rank"],
                "citation_correct": c["citation_correct"],
                "_fail_sort": 0 if not (c["recall_10"] and c["citation_correct"]) else 1,
            })
        df = pd.DataFrame(rows).sort_values("_fail_sort").drop(columns="_fail_sort")
        st.dataframe(df, width="stretch", hide_index=True)

    st.divider()

    # --- 1d. Chạy lại eval — chỉ hướng dẫn tĩnh, KHÔNG chạy subprocess thật trong UI
    # (rủi ro cao: block UI ~20 phút, tốn tiền thật nếu bấm nhầm, không kiểm soát được
    # tiến trình — task cho phép bỏ qua nếu khó implement an toàn).
    st.subheader("🔄 Chạy lại eval")
    st.info(
        "Chạy `python evals/run_evals.py` từ terminal để cập nhật số liệu trên "
        "(tốn ~$0.29, mất ~20-30 phút, gọi API thật)."
    )


@st.dialog("📊 Đánh giá hệ thống", width="large")
def eval_dashboard_dialog():
    render_eval_dashboard()


def _classify_answerability(answer: str) -> str:
    """
    Heuristic đơn giản, tái sử dụng REFUSAL_PATTERNS đã có trong evals/run_evals.py
    (không viết lại logic từ chối mới).
    """
    answer_lower = answer.lower()
    if any(re.search(p, answer_lower) for p in REFUSAL_PATTERNS):
        return "Từ chối"
    return "Trả lời"


def _word_overlap_ratio(answer: str, chunk_text: str) -> float:
    """Tỷ lệ % từ có nghĩa (>=3 ký tự) trong chunk xuất hiện trong answer.
    Heuristic thô — KHÔNG phải semantic similarity, chỉ đo lexical overlap.
    Dùng để cảnh báo nhanh, không phải bằng chứng chính xác: answer diễn đạt
    lại bằng từ khác vẫn có thể đúng nhưng overlap thấp."""
    def _tokenize(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\w+", s) if len(w) >= 3}

    chunk_words = _tokenize(chunk_text)
    if not chunk_words:
        return 0.0
    answer_words = _tokenize(answer)
    return len(chunk_words & answer_words) / len(chunk_words)


def render_trace_expander(trace_info: dict):
    with st.expander("🔍 Xem chi tiết xử lý (Agent Trace)"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Số bước lặp", trace_info["iterations"])
        col2.metric("Thời gian", f"{trace_info['latency_ms']:.0f} ms")
        col3.metric("Chi phí ước tính", f"${trace_info['estimated_cost_usd']:.6f}")

        # Tính SỚM (trước khi liệt kê chunk) để biết chunk nào thực sự được trích,
        # dùng chung cho cả preview dài hơn (bên dưới) lẫn phần "Đánh giá câu trả lời".
        answer = trace_info.get("answer", "")
        has_citation_marker = any(
            marker in answer for marker in ("Nguồn:", "trang", "p.")
        )

        search_chunks = []
        if trace_info["trace"]:
            st.write("**Công cụ đã gọi:**")
            for t in trace_info["trace"]:
                status_icon = "✅" if t["result"]["ok"] else "❌"
                st.write(f"{status_icon} Bước {t['iteration']}: `{t['tool_name']}`")
                st.json(t["arguments"])
                if not t["result"]["ok"]:
                    st.error(t["result"]["error"])
                elif t["tool_name"] == "search_knowledge_base":
                    for r in t["result"]["data"].get("results", []):
                        search_chunks.append(r)
                        is_cited = has_citation_marker and r["source"].lower() in answer.lower()
                        preview_len = 300 if is_cited else 100
                        preview = r["text"][:preview_len] + ("…" if len(r["text"]) > preview_len else "")
                        line = f"📄 {r['source']} (trang {r['page']}) · distance {r['distance']} — \"{preview}\""
                        margin = SCORE_THRESHOLD - r["distance"]
                        if margin < MARGIN_WARNING_THRESHOLD:
                            st.warning(f"{line}\n\n⚠️ Sát ngưỡng SCORE_THRESHOLD={SCORE_THRESHOLD} "
                                       f"(margin={margin:.4f})")
                        else:
                            st.caption(line)
        else:
            st.write("_Không có công cụ nào được gọi (trả lời trực tiếp từ model)._")

        if trace_info.get("usage_per_call"):
            st.write("**Token mỗi lần gọi LLM:**")
            st.json(trace_info["usage_per_call"])

        # --- Phần 2: Đánh giá câu trả lời (live eval per-question) ---
        st.divider()

        answerability = _classify_answerability(answer)

        with st.container(border=True):
            st.write("**📝 Đánh giá nhanh câu trả lời**")
            if search_chunks:
                matches_retrieved_source = any(
                    c["source"].lower() in answer.lower() for c in search_chunks
                )
                # Overlap từ vựng chỉ tính cho chunk THỰC SỰ được trích (has_citation_marker
                # + tên nguồn khớp) — lấy MAX nếu có nhiều chunk được trích (1 chunk khớp tốt
                # là đủ bằng chứng câu trả lời có bám nội dung gốc, không cần TB cả nhóm).
                cited_chunks = [
                    c for c in search_chunks
                    if has_citation_marker and c["source"].lower() in answer.lower()
                ]
                overlap_ratio = max(
                    (_word_overlap_ratio(answer, c["text"]) for c in cited_chunks), default=0.0
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trạng thái", answerability)
                c2.metric("Dấu hiệu trích dẫn", "Có" if has_citation_marker else "Không")
                c3.metric("Nguồn khớp answer", "Có" if matches_retrieved_source else "Không")
                c4.metric("Liên quan từ vựng", f"{overlap_ratio*100:.0f}%")
            else:
                st.metric("Trạng thái", answerability)
            st.caption(
                "_Kiểm tra nhanh (heuristic), không phải xác nhận chính thức. "
                "Đối chiếu chính xác (gold_evidence) chỉ có trong tab Đánh giá hệ thống._"
            )

        if search_chunks and has_citation_marker and overlap_ratio < 0.15:
            st.warning("⚠️ Answer trích nguồn nhưng ít từ trùng với nội dung chunk "
                       "— có thể agent diễn giải quá xa nội dung gốc, nên tự kiểm tra lại.")


def render_chat():
    st.title("⚡ Electricity Consumption AI Agent")
    st.caption("Hỏi bằng tiếng Việt về dữ liệu tiêu thụ điện hoặc tra cứu tài liệu EVN.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "trace_info" in msg:
                render_trace_expander(msg["trace_info"])

    with st.bottom:
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
                            "usage_per_call": result["usage_per_call"],
                            "answer": result["answer"],
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
    render_chat()
    render_sidebar()


if __name__ == "__main__":
    main()
