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
- Cross-encoder reranker: thử nghiệm, reject vì case xfail #11 → #18 (tệ hơn)
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
- Docker verified: upload dataset + câu hỏi theo tháng + RAG + memory + CSV lạ + ingest KB
  + xóa KB
