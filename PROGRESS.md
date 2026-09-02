# PROGRESS.md — Electricity Consumption AI Agent

## Tuần 1 (2026-08-26 → 2026-08-27)

### Đã hoàn thành
- Nghiên cứu + chọn LLM (so sánh Qwen3.6-27b vs DeepSeek V3.2, chọn Qwen dựa trên test
  hallucination resistance tiếng Việt)
- Thiết kế kiến trúc agent loop (observe → act → max_steps)
- Xây dựng 6 tool Pandas (aggregate_metric, group_metric, find_peak, compare_periods,
  calculate_correlation, get_dataset_summary, calculator)
- Xây dựng RAG pipeline: PyMuPDF extract → chunk → text-embedding-3-small → ChromaDB
- Nạp 17 nguồn KB (2 EVN + 1 Chỉ thị 20 + 14 IEA reports, 6.127 chunks)
- Streamlit UI: upload dataset CSV/Excel, chat, Agent Trace
- Phát hiện + fix bug date-range filtering (tồn tại từ M2, cắt 143 dòng ngày cuối)
- Phát hiện + fix empty-content glitch (~6-7% lượt gọi API)

### Vấn đề gặp phải
- IEA content che khuất EVN khi mở rộng KB 62 → 6.127 chunks → giải quyết bằng metadata
  filtering theo `document_type`
- Cross-lingual gap: câu tiếng Việt cho distance cao hơn 0.3-0.5 so với câu tiếng Anh cùng
  nội dung → ghi nhận, chưa giải quyết triệt để (để V2)

### Quyết định kỹ thuật có bằng chứng
- Hybrid BM25+RRF: thử nghiệm, reject vì 1 case tệ hơn (không đạt tiêu chí "không được tệ hơn")
- Cross-encoder reranker: đã thử, không đạt hiệu quả — chi tiết số liệu đang được khôi phục lại
- Threshold `SCORE_THRESHOLD=1.3`: đo thật, margin 0.41

## Tuần 2 (2026-08-28 → 2026-09-01)

### Đã hoàn thành
- Xây dựng eval suite: 53 test cases, 8 categories → 51/53 = 96.2%
- Upload KB động qua UI (thêm/xóa/liệt kê tài liệu)
- Docker deployment (build 1.39GB, verify end-to-end)
- Fix file_uploader infinite-rerun bug (chat input bị chặn sau upload dataset)
- Fix `derive_time_columns` tổng quát (CSV lạ hoạt động, auto-detect datetime)
- Fix `REFUSAL_INDICATORS` → regex patterns (giảm false-negative eval)
- Thêm log chi phí/token/duration khi ingest KB (UI + terminal)
- Chặn upload trùng lặp KB (prevent orphan chunks)
- Phát hiện + dọn 619 chunk orphan trong ChromaDB
- COST.md: 3 kịch bản, 3 đề xuất tối ưu, 3 rủi ro
- DATA_SOURCES.md: 10+ Known Limitations có bằng chứng

### Vấn đề gặp phải
- Orphan chunks do khác biệt source_id giữa script ingest (đặt tay) và UI upload
  (auto-slugify) → đã fix bằng chặn trùng lặp + dọn thủ công
- Streamlit `st.rerun()` xóa message vừa render → fix bằng session_state persist

### Kết quả cuối cùng
- 51/53 = 96.2% accuracy, $0.31/lần eval, 21.4s trung bình/câu
- 2 case fail còn lại: both_007 + parametric_001 (search-until-exhausted, đã ghi nhận,
  không phải bug retrieval)
- Docker verified: upload dataset + câu hỏi theo tháng + RAG + ingest KB + xóa KB
- Memory (nhớ ngữ cảnh phiên): code chạy đúng — `agent/agent.py` (`self.messages` tích lũy
  qua các lượt) + `agent/loop.py` (mutate cùng list reference qua mỗi `ask()`) — nhưng CHƯA
  có test/case xác nhận đúng ý nghĩa multi-turn (câu sau tham chiếu câu trước), và CHƯA
  verify riêng trong Docker
- CSV lạ trong Docker (2026-09-01, log: `logs/csv_docker_test_20260901.log`, force-added dù
  `logs/` bị gitignore, đã quét sạch không có key/token nào bị log): test qua AppTest mô
  phỏng `st.file_uploader` thật trên `app.py` chạy trong container (`docker-compose up`),
  schema hoàn toàn khác Tetouan (cột `ngay_ghi_nhan` định dạng `DD/MM/YYYY`, cột categorical
  `ma_tram`/`trang_thai`). KHÔNG crash ở bất kỳ bước nào (app, 3 tool gọi trực tiếp, Agent
  thật qua LLM) — nhưng phát hiện lỗi thật: `derive_time_columns()` không nhận diện được cột
  ngày `DD/MM/YYYY` (thiếu `dayfirst=True`, tỷ lệ parse rơi dưới ngưỡng 95% vì ngày >12 như
  15, 20, 25 bị coerce thành NaT khi hiểu theo MM/DD mặc định).
  **ĐÃ FIX (cùng ngày):** `_parse_datetime_best_effort()` mới trong `tools/dataset_manager.py`
  thử cả `dayfirst=False` và `dayfirst=True` cho mỗi cột ứng viên, chọn tỷ lệ parse cao hơn —
  không hardcode `dayfirst=True` cho toàn hàm (tránh phá cột ISO/US format vốn đã đúng).
  **Test regression:** `test_dmy_slash_format_detected_via_dayfirst` (tests/test_derive_time_columns.py)
  — xác nhận FAIL trên code cũ (`git stash` tạm bỏ fix, chạy test → fail đúng như bug thật),
  PASS sau khi khôi phục fix. Toàn bộ 7/7 test derive_time_columns + 8/8 test_dataset_manager
  (bao gồm cột `datetime` ISO của Tetouan) đều pass — 0 regression. Full suite: 104 passed,
  1 xfailed (từ 103 passed trước đó, +1 test mới, không mất test nào).

## Cuối tuần 1

**Repo / link code:**
https://github.com/hoangyen918/electricity-consumption-ai-agent

**Đã làm được (loop / tool / RAG / UI):**
Agent loop (observe → act → max_steps) hoạt động với 6 tool Pandas; RAG pipeline đầy đủ
(PyMuPDF extract → chunk → text-embedding-3-small → ChromaDB, 17 nguồn/6.127 chunks); UI
Streamlit có upload dataset, chat, và Agent Trace.

**Đã đọc / tham khảo (link):**
`dataquestio/tutorials/evaluating-llm-outputs` — tham khảo phương pháp LLM-as-a-Judge
(nguyên văn ghi trong README.md, không có URL đầy đủ trong repo để dẫn link chính xác)

**Kẹt / cần mentor:**
[intern tự bổ sung] — không tìm thấy ghi chú nào trong repo

**Tuần 2 dự định:**
Hoàn thiện eval suite (53 câu, 8 category), triển khai Docker, mở rộng tính năng KB
(upload/xóa động), viết đầy đủ COST.md + DATA_SOURCES.md.

## Trước demo tuần 2

**Cách chạy (1–2 dòng):**
Clone → venv → `pip install -r requirements.txt` → điền `.env` → `python scripts/prepare_data.py`
→ `python scripts/ingest_kb.py` (1 lần) → `streamlit run app.py`.

**URL hoặc Docker:**
Docker — `docker-compose up -d --build` (port 8501, theo `docker-compose.yml`)

**Sample CSV + tài liệu KB nằm ở:**
- CSV: `data/sample/tetouan_sample.csv`
- KB: `kb/raw/` (17 file PDF — EVN, IEA, Chỉ thị 20)

**Tự đánh giá % hoàn thành đề:** ___% *(intern tự điền)*
