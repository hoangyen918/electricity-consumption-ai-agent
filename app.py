"""Streamlit UI cho Electricity Consumption AI Agent."""
import json
import re
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tools.dataset_manager import DatasetManager
from tools.data_cleaning_tools import clean_missing_values, detect_duplicates
from tools.turbine_diagnostic_tools import (
    diagnose_theoretical_power_deviation, OPST_LABELS, calculate_turbine_kpis,
    _detect_site, _power_wind_columns, CLASSIFICATION_LABELS,
)
from agent.turbine_tool_wrappers import (
    group_anomaly_windows_for_agent, get_or_compute_power_curve, get_or_compute_classification,
)
from agent.agent import Agent
from config.settings import SETTINGS
from rag.vector_store import count_documents
from rag.retriever import SCORE_THRESHOLD
from evals.run_evals import REFUSAL_PATTERNS, THRESHOLDS
from evals.llm_judge import diagnose_rag_quality

st.set_page_config(page_title=SETTINGS.app_title, page_icon="⚡", layout="wide")

DEFAULT_DATASET_PATH = "data/processed/tetouan_clean.csv"
BASELINE_PATH = Path("evals/baselines/v1_baseline.json")
CASES_PATH = Path("evals/cases.jsonl")
MARGIN_WARNING_THRESHOLD = 0.1  # cảnh báo nếu distance cách SCORE_THRESHOLD dưới 0.1

# 5 file turbine gió Phase 1 — KHÔNG git-track (xem .gitignore), chỉ tồn tại nếu người
# dùng đã tự copy vào máy. Dashboard phải tự phát hiện thiếu file, không giả định có sẵn.
WIND_TURBINE_DATASET_PATHS = {
    "his_data_GT1": "data/wind_turbine/data1/his_data_GT1.csv",
    "WindFarm04_WT1": "data/wind_turbine/data2/WindFarm04_WT1.csv",
    "WindFarm04_WT2": "data/wind_turbine/data2/WindFarm04_WT2.csv",
    "WindFarm04_WT3": "data/wind_turbine/data2/WindFarm04_WT3.csv",
    "WindFarm04_WT4": "data/wind_turbine/data2/WindFarm04_WT4.csv",
}
# Nguồn: tercile WindSpd đo được trong investigation Phase 2 (không tính lại ở đây,
# xem báo cáo "Phân tích deviation trên nhóm vi phạm" — corr(deviation, WindSpd)=-0.29).
GT1_DEVIATION_TERCILE_WINDSPD = {"Thấp (~4.0 m/s)": 23.77, "Vừa (~6.2 m/s)": 17.74, "Cao (~8.6 m/s)": 15.12}


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

    st.session_state.setdefault("session_total_cost_usd", 0.0)

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

        total_cost = st.session_state.session_total_cost_usd
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
            st.session_state.session_total_cost_usd = 0.0
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


def _render_rag_judge_section(trace_info: dict, search_chunks: list[dict], msg_id) -> None:
    """Nút 'Chấm chi tiết' — CHỈ gọi Judge khi bấm, kết quả cache theo msg_id trong
    session_state để rerun/bấm lại KHÔNG gọi lại API (tốn tiền + chậm 15-20s)."""
    cache = st.session_state.setdefault("rag_judge_cache", {})

    if st.button("🔍 Chấm chi tiết (Retrieval/Generation)", key=f"rag_judge_btn_{msg_id}"):
        if msg_id not in cache:
            with st.spinner("Đang chấm Retrieval/Generation bằng LLM Judge (~15-20s)..."):
                cache[msg_id] = diagnose_rag_quality(
                    question=trace_info.get("question", ""),
                    answer=trace_info.get("answer", ""),
                    retrieved_chunks=search_chunks,
                )

    if msg_id in cache:
        _display_rag_judge_result(cache[msg_id])


def _display_rag_judge_result(result: dict) -> None:
    verdict = result.get("verdict")
    verdict_display = {
        "OK": ("✅ OK — không phát hiện vấn đề rõ rệt", st.success),
        "RETRIEVAL_ISSUE": ("🔴 Vấn đề ở RETRIEVAL — chunk lấy về không liên quan tới câu hỏi", st.error),
        "GENERATION_ISSUE": ("🔴 Vấn đề ở GENERATION — câu trả lời không bám sát nội dung chunk", st.error),
        "BOTH_ISSUES": ("🔴 Vấn đề CẢ HAI — retrieval kém VÀ câu trả lời không bám chunk", st.error),
        "NO_CONTEXT": ("⚪ Không có chunk nào để đánh giá retrieval", st.info),
        "JUDGE_ERROR": ("⚠️ Judge gọi API lỗi — thử bấm lại", st.warning),
    }
    label, renderer = verdict_display.get(verdict, (f"Kết quả: {verdict}", st.info))
    renderer(label)

    faithfulness = result.get("faithfulness")
    relevancy = result.get("answer_relevancy")
    context_prec = result.get("context_precision")

    c1, c2, c3 = st.columns(3)
    c1.metric("Faithfulness", f"{faithfulness['faithfulness_score']*100:.0f}%" if faithfulness else "N/A")
    c2.metric("Answer Relevancy", f"{relevancy['relevancy_score']*100:.0f}%" if relevancy else "N/A")
    context_score = context_prec.get("context_precision_score") if context_prec else None
    c3.metric("Context Precision", f"{context_score*100:.0f}%" if context_score is not None else "N/A")

    st.caption(
        "_Ngưỡng heuristic ban đầu (chưa validate tuyệt đối): context_precision<0.5 "
        "và faithfulness>=0.8 → RETRIEVAL_ISSUE; context_precision>=0.5 và faithfulness<0.8 "
        "→ GENERATION_ISSUE. Context Precision = Average Precision (AP@k) trên verdict "
        "true/false của Judge cho từng chunk, có trọng số theo rank — không phải tỷ lệ đơn giản._"
    )

    if faithfulness and faithfulness.get("claims"):
        unsupported = [c for c in faithfulness["claims"] if not c.get("supported")]
        if unsupported:
            with st.expander(f"❌ {len(unsupported)} claim không có căn cứ trong chunk"):
                for c in unsupported:
                    st.write(f"- **{c.get('text', '')}** — {c.get('reason', '')}")

    if context_prec and context_prec.get("chunks"):
        chunks_detail = context_prec["chunks"]
        with st.expander(f"📄 Chi tiết {len(chunks_detail)} chunk (verdict retrieval)"):
            for i, c in enumerate(chunks_detail, start=1):
                mark = "✅" if c.get("relevant") else "❌"
                st.write(f"{mark} [{i}] **{c.get('source', '')}** (trang {c.get('page', '')}) — relevant={c.get('relevant')}")
            if context_prec.get("explanation"):
                st.caption(context_prec["explanation"])


def render_trace_expander(trace_info: dict, msg_id):
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

        if search_chunks:
            st.divider()
            _render_rag_judge_section(trace_info, search_chunks, msg_id)


def _get_wind_turbine_manager() -> DatasetManager:
    """DatasetManager RIÊNG cho dashboard turbine gió — tách biệt với
    st.session_state.dataset_manager của chat, tránh ảnh hưởng tới agent/chat_history
    khi người dùng chỉ đang xem dashboard."""
    if "wind_turbine_manager" not in st.session_state:
        st.session_state.wind_turbine_manager = DatasetManager()
    return st.session_state.wind_turbine_manager


def _ensure_wind_turbine_dataset_loaded(manager: DatasetManager, key: str) -> bool:
    """Load lazy (chỉ load lần đầu, cache trong manager) 1 trong 5 dataset turbine gió.
    Trả False + hiển thị lỗi rõ ràng nếu file không tồn tại (5 file này KHÔNG git-track,
    có thể thiếu trên máy người dùng chưa copy file — xem WIND_TURBINE_DATASET_PATHS)."""
    if key in manager.list_datasets():
        return True
    path = Path(WIND_TURBINE_DATASET_PATHS[key])
    if not path.exists():
        st.error(
            f"Không tìm thấy file `{path}`. 5 file turbine gió không được git-track "
            "(xem .gitignore) — cần tự copy vào đúng thư mục `data/wind_turbine/`."
        )
        return False
    with st.spinner(f"Đang nạp {key} ({path.stat().st_size / 1e6:.1f} MB)..."):
        manager.load_csv(str(path), name=key)
    return True


def _render_overview_tab():
    st.subheader("📋 Tổng quan dataset turbine gió")
    manager = _get_wind_turbine_manager()
    selected = st.selectbox("Chọn dataset:", list(WIND_TURBINE_DATASET_PATHS.keys()), key="overview_dataset_select")

    if not _ensure_wind_turbine_dataset_loaded(manager, selected):
        return

    df = manager.get_dataframe(selected)
    report = clean_missing_values(manager, dataset_name=selected, strategy="report")
    dup = detect_duplicates(manager, dataset_name=selected)

    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng", f"{len(df):,}")
    if report["ok"]:
        total_missing_cells = sum(v["count"] for v in report["data"]["missing_by_column"].values())
        total_cells = len(df) * len(df.columns)
        col2.metric("Tỷ lệ missing (toàn dataset)", f"{100*total_missing_cells/total_cells:.2f}%")
    else:
        col2.error(report["error"])
    if dup["ok"]:
        col3.metric("Dòng trùng lặp", f"{dup['data']['duplicate_count']:,}")
    else:
        col3.error(dup["error"])

    if report["ok"] and report["data"]["missing_by_column"]:
        st.write("**Chi tiết missing theo cột (chỉ cột có >0):**")
        rows = [
            {"cột": col, "số dòng thiếu": v["count"], "% thiếu": v["pct"]}
            for col, v in report["data"]["missing_by_column"].items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    elif report["ok"]:
        st.success("Không có cột nào bị missing trong dataset này.")


def _render_categorical_comparison_chart(field_name: str, field_data: dict):
    """Grouped bar chart (vi phạm vs bình thường) cho 1 field categorical
    ({"violating_pct": {...}, "normal_pct": {...}}). Field 'GT1.Common.OpSt' map key số
    sang nhãn chữ qua OPST_LABELS (nguồn tools/turbine_diagnostic_tools.py); field khác
    (hour, month) giữ nguyên key số."""
    violating_pct = field_data["violating_pct"]
    normal_pct = field_data["normal_pct"]
    all_keys = sorted(set(violating_pct.keys()) | set(normal_pct.keys()), key=lambda k: float(k))

    if field_name == "GT1.Common.OpSt":
        x_labels = [OPST_LABELS.get(int(float(k)), k) for k in all_keys]
    else:
        x_labels = all_keys

    fig = go.Figure(data=[
        go.Bar(name="Vi phạm", x=x_labels, y=[violating_pct.get(k, 0) for k in all_keys]),
        go.Bar(name="Bình thường", x=x_labels, y=[normal_pct.get(k, 0) for k in all_keys]),
    ])
    fig.update_layout(barmode="group", yaxis_title="% trong nhóm", xaxis_title=field_name, title=field_name)
    st.plotly_chart(fig, width="stretch")


def _render_diagnostic_tab():
    st.subheader("🔍 Diagnostic — vấn đề đã phát hiện")
    finding = st.selectbox(
        "Chọn vấn đề muốn xem:",
        ["TheoActiveW Deviation (GT1)"],  # danh sách se dai ra khi co finding moi
        key="diagnostic_finding_select",
    )

    if finding == "TheoActiveW Deviation (GT1)":
        manager = _get_wind_turbine_manager()
        if not _ensure_wind_turbine_dataset_loaded(manager, "his_data_GT1"):
            return

        result = diagnose_theoretical_power_deviation(manager, dataset_name="his_data_GT1")
        if not result["ok"]:
            st.error(result["error"])
            return

        data = result["data"]
        col1, col2 = st.columns(2)
        col1.metric("Số dòng vi phạm (>10% lý thuyết)", f"{data['violation_count']:,}")
        col2.metric("Tỷ lệ", f"{data['violation_pct']:.2f}%")

        st.write("**So sánh nhóm vi phạm vs bình thường:**")
        numeric_rows = []
        for field_name, field_data in data["comparison"].items():
            if "violating_pct" in field_data:
                _render_categorical_comparison_chart(field_name, field_data)
            elif "violating" in field_data:
                v, n = field_data["violating"], field_data["normal"]
                numeric_rows.append({
                    "Field": field_name,
                    "Vi phạm (mean±std)": f"{v['mean']:.2f} ± {v['std']:.2f}",
                    "Bình thường (mean±std)": f"{n['mean']:.2f} ± {n['std']:.2f}",
                })

        if numeric_rows:
            st.write("**Bảng so sánh các field liên tục:**")
            st.dataframe(pd.DataFrame(numeric_rows), width="stretch", hide_index=True)

        st.write("**Deviation% theo tercile WindSpd (đã đo trong investigation Phase 2, không tính lại):**")
        fig = px.bar(
            x=list(GT1_DEVIATION_TERCILE_WINDSPD.keys()),
            y=list(GT1_DEVIATION_TERCILE_WINDSPD.values()),
            labels={"x": "Tercile WindSpd", "y": "Deviation trung bình (%)"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Deviation% giảm dần khi WindSpd tăng — gợi ý sai số đo lường bị khuếch đại "
            "ở gió thấp (xem báo cáo investigation Phase 2), không phải turbine 'vượt trội' "
            "ở gió mạnh."
        )


def _render_kpi_summary_tab():
    st.subheader("📊 KPI Summary — Production / Losses / Capacity Factor / TBA / PBA")
    manager = _get_wind_turbine_manager()
    selected = st.selectbox("Chọn dataset:", list(WIND_TURBINE_DATASET_PATHS.keys()), key="kpi_dataset_select")

    if not _ensure_wind_turbine_dataset_loaded(manager, selected):
        return

    df = manager.get_dataframe(selected)
    site = _detect_site(df)
    if site is None:
        st.error("Không nhận diện được cấu trúc turbine gió (cần cột bắt đầu bằng 'GT1.' hoặc cả 'ACTIVE_POWER'/'WIND_SPEED') cho dataset này.")
        return

    power_column, wind_speed_column = _power_wind_columns(site)
    setpoint_column = "GT1.SP.ActualSPP" if site == "site1" else None
    theoretical_power_column = (
        "GT1.Common.TheoActiveW" if site == "site1" and "GT1.Common.TheoActiveW" in df.columns else None
    )

    # Dung get_or_compute_power_curve/get_or_compute_classification (TU agent/
    # turbine_tool_wrappers.py) — CUNG cache voi tab Power Curve/Timeline, thay vi goi
    # thang estimate_power_curve()/classify_turbine_status() (truoc day khien 3 tab
    # nay KHONG chia se cache, moi tab tu tinh lai rieng du CUNG dataset/cot/bin_width).
    with st.spinner("Đang tính power curve + phân loại trạng thái (lần đầu có thể mất vài giây)..."):
        curve_result = get_or_compute_power_curve(manager, selected, power_column, wind_speed_column, 1.0)
        if not curve_result["ok"]:
            st.error(curve_result["error"])
            return

        classify_result = get_or_compute_classification(
            manager, selected, power_column, wind_speed_column, setpoint_column, 1.0,
        )
        if not classify_result["ok"]:
            st.error(classify_result["error"])
            return

        result = calculate_turbine_kpis(
            manager, dataset_name=selected, power_column=power_column, wind_speed_column=wind_speed_column,
            classification=classify_result["data"]["classification"], power_curve=curve_result["data"],
            theoretical_power_column=theoretical_power_column,
        )
    if not result["ok"]:
        st.error(result["error"])
        return

    data = result["data"]
    method_label = "Trực tiếp (TheoActiveW)" if data["method"] == "direct_theoretical" else "Ước lượng (power curve)"
    st.caption(f"Reachable energy: **{method_label}** — Site 1 dùng cột lý thuyết trực tiếp (chính xác hơn), "
               f"Site 2 ước lượng qua power curve (kém chính xác hơn).")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Production (MWh)", f"{data['production_mwh']:,.1f}")
    col2.metric("Losses (%)", f"{data['losses_pct']:.2f}%" if data["losses_pct"] is not None else "N/A")
    col3.metric("Capacity Factor (%)", f"{data['capacity_factor_pct']:.2f}%" if data["capacity_factor_pct"] is not None else "N/A")
    col4.metric("TBA (%)", f"{data['tba_pct']:.2f}%" if data["tba_pct"] is not None else "N/A")
    col5.metric("PBA (%)", f"{data['pba_pct']:.2f}%" if data["pba_pct"] is not None else "N/A")

    if data.get("tba_note"):
        st.caption(f"ℹ️ {data['tba_note']}")

    excluded_note = (
        f" ({data['n_rows_excluded_no_reachable_estimate']:,} dòng bị loại do thiếu reachable estimate)"
        if data["n_rows_excluded_no_reachable_estimate"] else ""
    )
    st.caption(
        f"Rated power: {data['rated_power_kw']:,.1f} kW · Reachable energy: {data['reachable_mwh']:,.1f} MWh · "
        f"Interval: {data['interval_minutes']:.1f} phút · {data['n_rows_used']:,} dòng dùng để tính{excluded_note}"
    )


# Mau co dinh cho 8 nhan Meteodyn — dung chung cho ca Power Curve (scatter) va Timeline
# (Gantt bar). Normal mau xam nhat lam nen (chiem da so diem, khong can noi bat); cac
# nhan bat thuong dung mau tuoi/de phan biet. "Measurement Error" xam DAM HON Normal
# (khac han, tranh nham voi nen) vi day la loi du lieu, khong phai trang thai van hanh
# that.
_CLASSIFICATION_STATUS_COLORS = {
    "Normal": "#BEBEBE",
    "Curtailment": "#1f77b4",
    "Partial Curtailment": "#17becf",
    "Under Production": "#ff7f0e",
    "Over Production": "#9467bd",
    "Stop": "#d62728",
    "Partial Stop": "#e377c2",
    "Measurement Error": "#7f7f7f",
}

_POWER_CURVE_SAMPLE_SIZE = 8000


def _render_power_curve_tab():
    st.subheader("📈 Power Curve — Wind Speed vs Power, tô màu theo trạng thái")
    manager = _get_wind_turbine_manager()
    selected = st.selectbox("Chọn dataset:", list(WIND_TURBINE_DATASET_PATHS.keys()), key="powercurve_dataset_select")

    if not _ensure_wind_turbine_dataset_loaded(manager, selected):
        return

    df = manager.get_dataframe(selected)
    site = _detect_site(df)
    if site is None:
        st.error("Không nhận diện được cấu trúc turbine gió cho dataset này.")
        return

    power_column, wind_speed_column = _power_wind_columns(site)
    setpoint_column = "GT1.SP.ActualSPP" if site == "site1" else None

    # Dung get_or_compute_power_curve/get_or_compute_classification (TU agent/
    # turbine_tool_wrappers.py, CUNG cache tren instance DatasetManager ma
    # group_anomaly_windows_for_agent — tab Timeline — dung) THAY VI goi thang
    # estimate_power_curve()/classify_turbine_status() — phat hien that qua AppTest:
    # goi truc tiep (khong cache) khien MOI lan Streamlit rerun (vd doi widget o BAT
    # KY tab nao, vi st.tabs() chay lai TOAN BO than cua ca 5 tab) deu tinh lai
    # classify_turbine_status TU DAU cho dataset dang chon o tab nay — cong don voi
    # cac tab nang khac (KPI Summary, Timeline) gay timeout that (>180s) khi nhieu tab
    # dang chon dataset lon khac nhau cung luc. Dung ham cache chung -> neu tab khac
    # (vd Timeline) da tinh SAN cho DUNG dataset/cot/bin_width nay, tab nay TAI DUNG
    # ngay, khong tinh lai.
    with st.spinner("Đang tính power curve + phân loại trạng thái (lần đầu có thể mất vài giây)..."):
        curve_result = get_or_compute_power_curve(manager, selected, power_column, wind_speed_column, 1.0)
        if not curve_result["ok"]:
            st.error(curve_result["error"])
            return

        classify_result = get_or_compute_classification(
            manager, selected, power_column, wind_speed_column, setpoint_column, 1.0,
        )
        if not classify_result["ok"]:
            st.error(classify_result["error"])
            return

    # classification la {row_index: label} (da doc ky code that, khong doan) — align
    # voi df.index qua reindex, drop cac dong khong co nhan (khong nen xay ra thuc te
    # vi classify_turbine_status() gan nhan cho MOI dong, phong ngua thoi).
    labels = pd.Series(classify_result["data"]["classification"]).reindex(df.index)
    plot_df = pd.DataFrame({
        "wind_speed": df[wind_speed_column],
        "power": df[power_column],
        "label": labels,
    }).dropna(subset=["label"])

    total_points = len(plot_df)
    if total_points > _POWER_CURVE_SAMPLE_SIZE:
        plot_df = plot_df.sample(n=_POWER_CURVE_SAMPLE_SIZE, random_state=42)
        st.caption(
            f"Hiển thị mẫu ngẫu nhiên {_POWER_CURVE_SAMPLE_SIZE:,} điểm trên tổng "
            f"{total_points:,} điểm (tránh render chậm với dataset lớn)."
        )

    # Ve Normal TRUOC (nen), cac nhan bat thuong SAU — de nhan bat thuong (thieu so)
    # khong bi che khuat boi khoi diem Normal (da so).
    labels_present = set(plot_df["label"].unique())
    ordered_labels = [l for l in CLASSIFICATION_LABELS if l in labels_present]
    ordered_labels.sort(key=lambda l: 0 if l == "Normal" else 1)

    fig = go.Figure()
    for label in ordered_labels:
        subset = plot_df[plot_df["label"] == label]
        fig.add_trace(go.Scatter(
            x=subset["wind_speed"], y=subset["power"], mode="markers", name=label,
            marker=dict(size=4, color=_CLASSIFICATION_STATUS_COLORS.get(label, "#000000")),
            opacity=0.6,
        ))

    # Duong power curve (median moi bin) ve DE LEN lam duong tham chieu — line, khong
    # phai marker, de phan biet ro voi cac diem du lieu that ben duoi.
    curve_bins = curve_result["data"]["power_curve"]
    curve_x = [b["wind_speed_bin_center"] for b in curve_bins if b["median_power"] is not None]
    curve_y = [b["median_power"] for b in curve_bins if b["median_power"] is not None]
    fig.add_trace(go.Scatter(
        x=curve_x, y=curve_y, mode="lines", name="Power Curve (median)",
        line=dict(color="black", width=2),
    ))

    fig.update_layout(
        xaxis_title="Wind Speed (m/s)", yaxis_title=f"Power ({power_column})",
        title=f"Power Curve — {selected}", legend_title="Trạng thái",
    )
    st.plotly_chart(fig, width="stretch")


def _render_timeline_tab():
    st.subheader("🕐 Timeline các khoảng thời gian bất thường")
    manager = _get_wind_turbine_manager()
    selected = st.selectbox("Chọn dataset:", list(WIND_TURBINE_DATASET_PATHS.keys()), key="timeline_dataset_select")

    if not _ensure_wind_turbine_dataset_loaded(manager, selected):
        return

    df = manager.get_dataframe(selected)
    site = _detect_site(df)
    if site is None:
        st.error("Không nhận diện được cấu trúc turbine gió cho dataset này.")
        return
    setpoint_column = "GT1.SP.ActualSPP" if site == "site1" else None

    # Mac dinh 30 ngay GAN NHAT theo du lieu THAT cua dataset (khong phai hom nay) —
    # tranh xu ly toan bo 2-5 nam cung luc (cham + qua nhieu window de doc).
    data_min = df["datetime"].min().date()
    data_max = df["datetime"].max().date()
    default_start = max(data_min, data_max - pd.Timedelta(days=30))
    # key PHAI rieng theo TUNG dataset (khong dung 1 key co dinh chung) — phat hien that
    # qua AppTest: cac dataset co khoang ngay HOAN TOAN khac nhau (GT1: 2023-2025, WT1:
    # 2012-2016), neu dung chung 1 key, widget giu lai gia tri CU (tu dataset truoc) khi
    # doi dataset, gia tri do co the nam NGOAI min_value/max_value MOI -> Streamlit xu
    # ly sai trang thai, gay treo (>180s) khi test qua AppTest (dashboard day du 5 tab).
    date_range = st.date_input(
        "Khoảng thời gian:", value=(default_start, data_max),
        min_value=data_min, max_value=data_max, key=f"timeline_date_range_{selected}",
    )
    if len(date_range) != 2:
        st.info("Chọn đủ 2 mốc ngày (bắt đầu và kết thúc).")
        return
    start_date, end_date = date_range
    if start_date > end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        return

    with st.spinner("Đang phân loại trạng thái + gộp các khoảng thời gian bất thường..."):
        # group_anomaly_windows_for_agent (KHÔNG phải group_anomaly_windows gốc) — CHỈ
        # bản wrapper này có start_date/end_date (fix cap-window tuần trước, xem
        # agent/turbine_tool_wrappers.py). BẮT BUỘC truyền start_date/end_date ở đây vì
        # đã giới hạn khoảng thời gian — nếu không, cap mặc định 30-khoảng-dài-nhất-TOÀN-
        # BỘ-DATASET sẽ áp dụng SAI phạm vi và có thể bỏ sót/trộn window ngoài khoảng.
        windows_result = group_anomaly_windows_for_agent(
            manager, dataset_name=selected, timestamp_column="datetime",
            power_column=None, wind_speed_column=None, setpoint_column=setpoint_column,
            bin_width=1.0, min_consecutive_points=3, exclude_labels=None,
            start_date=str(start_date), end_date=str(end_date),
        )
    if not windows_result["ok"]:
        st.error(windows_result["error"])
        return

    windows = windows_result["data"]["windows"]
    if not windows:
        st.info(f"Không có khoảng thời gian bất thường nào trong {start_date} → {end_date} (toàn bộ Normal).")
        return

    timeline_df = pd.DataFrame(windows)
    timeline_df["start_time"] = pd.to_datetime(timeline_df["start_time"])
    timeline_df["end_time"] = pd.to_datetime(timeline_df["end_time"])

    fig = px.timeline(
        timeline_df, x_start="start_time", x_end="end_time", y="label", color="label",
        color_discrete_map=_CLASSIFICATION_STATUS_COLORS,
        hover_data=["n_points", "duration"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"Timeline bất thường — {selected} ({start_date} → {end_date})",
        xaxis_title="Thời gian", legend_title="Trạng thái",
    )
    st.plotly_chart(fig, width="stretch")

    truncated_note = ""
    if windows_result["data"].get("windows_truncated"):
        truncated_note = (
            f" (đã cắt bớt — chỉ hiển thị {len(windows)} khoảng DÀI NHẤT trong tổng "
            f"{windows_result['data']['windows_total_found']} khoảng tìm được trong khoảng thời gian này)"
        )
    st.caption(
        f"{len(windows)} khoảng thời gian bất thường (đã loại nhãn 'Normal', chỉ gộp "
        f"khoảng có ≥3 điểm liên tiếp cùng nhãn){truncated_note}."
    )


def render_wind_turbine_dashboard():
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Tổng quan", "📈 Power Curve", "🕐 Timeline",
        "🔍 Diagnostic", "📊 KPI Summary",
    ])
    with tab1:
        _render_overview_tab()
    with tab2:
        _render_power_curve_tab()
    with tab3:
        _render_timeline_tab()
    with tab4:
        _render_diagnostic_tab()
    with tab5:
        _render_kpi_summary_tab()


def render_chat():
    st.title("⚡ Electricity Consumption AI Agent")
    st.caption("Hỏi bằng tiếng Việt về dữ liệu tiêu thụ điện hoặc tra cứu tài liệu EVN.")

    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "trace_info" in msg:
                render_trace_expander(msg["trace_info"], idx)

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
                        st.session_state.session_total_cost_usd += result["estimated_cost_usd"]
                        st.write(result["answer"])
                        trace_info = {
                            "iterations": result["iterations"],
                            "latency_ms": result["latency_ms"],
                            "estimated_cost_usd": result["estimated_cost_usd"],
                            "trace": result["trace"],
                            "usage_per_call": result["usage_per_call"],
                            "answer": result["answer"],
                            "question": question,
                        }
                        # msg_id = index NÓ sẽ có trong chat_history sau khi append bên dưới
                        # (user message đã append, len hiện tại == index tương lai của assistant)
                        render_trace_expander(trace_info, len(st.session_state.chat_history))
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
    mode = st.sidebar.radio(
        "Chế độ xem", ["💬 Chat", "📊 Wind Turbine Dashboard"], key="view_mode"
    )
    if mode == "💬 Chat":
        # render_chat() PHẢI chạy trước render_sidebar() (giữ nguyên fix cost-display-lag
        # trước đây — sidebar đọc session_total_cost_usd, nếu chạy trước render_chat()
        # trong cùng 1 rerun sẽ hiển thị số liệu trễ 1 câu hỏi). Lệch so với thứ tự
        # render_sidebar() trước trong task gốc — cố ý giữ đúng thứ tự đã fix.
        render_chat()
        render_sidebar()
    else:
        # Nhánh nay KHONG BAO GIO goi render_chat() — dashboard co st.tabs ben trong,
        # an toan vi chat_input (trong render_chat()) khong bi long vao tabs.
        render_sidebar()
        render_wind_turbine_dashboard()


if __name__ == "__main__":
    main()
