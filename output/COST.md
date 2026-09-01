# Báo cáo chi phí — Electricity Consumption AI Agent

## Nguồn giá
- Chat model (qwen/qwen3.6-27b qua OpenRouter): $0.30/1M input tokens, $2.00/1M output tokens
  (giá niêm yết openrouter.ai/qwen/qwen3.6-27b). OpenRouter định tuyến qua nhiều nhà cung cấp,
  giá thực tế có thể dao động nhẹ ($0.289-0.35/1M input tùy provider).
- Embedding model (openai/text-embedding-3-small qua OpenRouter): $0.02/1M tokens.

## Chi phí đo được thực tế (số liệu chính thức, 29/08/2026)

### Bộ 53 câu eval — chạy đầy đủ
- Tổng chi phí: **$0.310503**
- Trung bình mỗi câu: **~$0.0059**
- Latency trung bình: **21.4 giây/câu**
- Accuracy: 96.2% (51/53)

### Ingest toàn bộ Knowledge Base (một lần duy nhất, không lặp lại trừ khi thêm tài liệu mới)
- 17 nguồn, 6,127 chunks, ~3.1 triệu token (embedding)
- Chi phí: dưới $0.10 (embedding rẻ, $0.02/1M token)

### 3 kịch bản sử dụng cụ thể (trích từ evals/results/eval_detail_20260829_210926.json)

**Kịch bản 1 — Data-only:** ví dụ "Zone 1 trung bình tháng 7" — 2-3 iteration, ~$0.003-0.005,
~10-15 giây.

**Kịch bản 2 — RAG-only:** ví dụ câu hỏi kb_qa — 2 iteration, ~$0.004-0.006, ~15-20 giây.

**Kịch bản 3 — BOTH mode:** ví dụ câu hỏi both_ — 3-5 iteration (đắt nhất, do phải tra cứu
rồi tính), ~$0.008-0.015, có thể lên 30-60 giây nếu agent thử nhiều lần tìm kiếm.

### Ước tính 1 phiên demo (10 câu hỏi hỗn hợp)
~$0.06 (10 × $0.0059 trung bình) ≈ **1.560 VND** (tỷ giá tham khảo 26.000 VND/USD)

## Đề xuất tối ưu chi phí

1. **Giới hạn số lần lặp search_knowledge_base không cải thiện** — đã phát hiện qua eval
   chính thức: 2/53 câu (both_007, parametric_001) tốn hết 5 iteration vì agent gọi
   search_knowledge_base 5 lần liên tiếp không dừng, tốn gấp ~3 lần chi phí bình thường
   cho câu đó. Thêm rule giới hạn 2-3 lần thử trước khi kết luận sẽ giảm chi phí đáng kể
   cho nhóm câu hỏi khó.
2. **Cache embedding câu hỏi lặp lại** — hiện mỗi câu hỏi luôn tính embedding mới.
3. **Batch ingest thông minh** — dùng hash nội dung để skip chunk không đổi khi re-ingest.

## Rủi ro chi phí cần lưu ý

1. **Corpus lớn (6,127 chunks) làm tăng token context** khi 1 tool trả về payload lớn
   (đã ghi nhận với group_metric ở M8).
2. **API key có hạn dùng, ngân sách cố định** (limit_reset: null) — theo dõi qua GET /api/v1/key.
3. **Giá OpenRouter dao động theo provider** được tự động chọn mỗi lần gọi.
4. **Pattern "search-until-exhausted"** (mục Đề xuất tối ưu #1) có thể làm chi phí trung bình
   tăng bất thường nếu tỷ lệ câu hỏi khó tăng lên trong thực tế sử dụng.
