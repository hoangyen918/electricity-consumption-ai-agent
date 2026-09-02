# Báo cáo chi phí — Electricity Consumption AI Agent

## Nguồn giá token
- Chat model — [qwen/qwen3.6-27b trên OpenRouter](https://openrouter.ai/qwen/qwen3.6-27b):
  $0.30/1M input tokens, $2.00/1M output tokens.
- Embedding model — [openai/text-embedding-3-small trên OpenRouter](https://openrouter.ai/openai/text-embedding-3-small):
  $0.02/1M tokens.
- **Ngày xác nhận giá: 2026-08-26** (nguồn: `PRICING_LAST_VERIFIED` trong `services/cost_tracker.py`).
- OpenRouter định tuyến qua nhiều nhà cung cấp, giá thực tế có thể dao động nhẹ
  ($0.289-0.35/1M input tùy provider). Đối chiếu chính xác tuyệt đối cho 1 request cụ thể:
  `GET https://openrouter.ai/api/v1/generation?id=<response_id>` (id lấy từ `response.id`).

## 1 phiên demo điển hình

Đo trực tiếp qua `agent.ask()` thật (2026-09-02), câu hỏi `data_001`
("Zone 1 trung bình tháng 7 là bao nhiêu?", dataset `data/processed/tetouan_clean.csv`):

| Hạng mục | Giá trị |
|----------|---------|
| Số bước loop | 2 |
| Lần gọi LLM | 2 |
| Token in / out (tổng) | 6.849 in / 624 out |
| Embedding (nếu có) | 0 (câu hỏi thuần data, không gọi `search_knowledge_base`) |
| **USD ước tính** | **$0.0033027** |

Lưu ý: tool thực tế được chọn ở lần đo này là `group_metric`. Case `data_001` trong eval
chính thức (`evals/results/eval_detail_20260901_011250.json`) từng dùng `get_dataset_summary`
+ `aggregate_metric` (3 iterations, $0.0044193) — khác lựa chọn tool do LLM không hoàn toàn
deterministic giữa các lần gọi (đã ghi nhận trong `DATA_SOURCES.md`), cả 2 cách đều ra đúng
kết quả.

### Ảnh hưởng của multi-dataset (thêm 2026-09-02, xem `logs/multi_dataset_docker_test_20260902.log`)

**Upload 2 file thay vì 1 — embedding cost:** **$0 chênh lệch.** Upload dataset CSV/Excel
(`DatasetManager.load_csv`/`load_excel`) chỉ đọc bằng `pandas`, KHÔNG gọi embedding model —
embedding chỉ phát sinh khi nạp tài liệu vào Knowledge Base (PDF/TXT/MD qua
`scripts/ingest_kb.py` hoặc UI "Thêm tài liệu vào KB"), không liên quan tới upload dataset.
Đã xác nhận qua đọc code `tools/dataset_manager.py` (không có lời gọi embedding nào).

**Câu hỏi cross-dataset — số bước/tool call tăng:** Đo thật qua log Docker (agent mới, không
có lịch sử hội thoại): câu hỏi trên **1** dataset (kịch bản d, "solar station trung bình")
chỉ tốn **1 tool call** (1 iteration). Câu hỏi **cross-dataset** (kịch bản e, "so sánh Zone 1
với solar station") tốn **4 iterations, 5 tool call** (`get_dataset_summary` → 2×
`aggregate_metric` với `dataset_name` khác nhau → 2× `calculate`).

Log không ghi lại token/USD cho phiên multi-dataset này (script test chỉ log tool call + câu
trả lời, không log cost) → **không có số đo thật cho $ ở đây**. Ước tính dựa trên dữ liệu cost
đã đo thật ở trên (case `data_001`, 2-3 iterations ≈ $0.0033–$0.0044): ngoại suy tuyến tính
theo tỷ lệ số bước lặp (4 bước cross-dataset / 1 bước đơn dataset ≈ 4×số bước, nhưng phần lớn
chi phí nằm ở 2 lời gọi LLM đầu đã có context dài, các bước `calculate` sau rẻ vì không cần
gọi lại `search`/tool nặng) → **ước tính khoảng 1.5–2.5 lần chi phí của 1 câu hỏi đơn
dataset**, tương đương **~$0.005–$0.008/câu cross-dataset** thay vì ~$0.003/câu đơn dataset.
**Đây là ước tính ngoại suy, chưa phải số đo trực tiếp** — cần đo thật qua `cost_tracker` nếu
cần số chính xác cho báo cáo chính thức.

## 3 kịch bản sử dụng

Trung bình theo category, tính từ **`evals/results/eval_detail_20260901_011250.json`**
(bộ eval chính thức 51/53, 2026-09-01 — thay cho file cũ `eval_detail_20260829_210926.json`
đã dùng trước đây):

| Kịch bản | USD / lần |
|----------|-----------|
| Câu hỏi data đơn giản (1–2 tool) | $0.003756 (TB 10 case `data_simple`, min $0.002273 – max $0.004832) |
| Câu hỏi RAG (retrieve + trả lời) | $0.005637 (TB 14 case `kb_qa`, min $0.002755 – max $0.008183) |
| Câu kết hợp data + knowledge | $0.011932 (TB 7 case `both`, min $0.006854 – max $0.028888) |

## Chi phí đo được thực tế (số liệu chính thức, 29/08/2026)

### Bộ 53 câu eval — chạy đầy đủ
- Tổng chi phí: **$0.310503**
- Trung bình mỗi câu: **~$0.0059**
- Latency trung bình: **21.4 giây/câu**
- Accuracy: 96.2% (51/53)

### Ingest toàn bộ Knowledge Base (một lần duy nhất, không lặp lại trừ khi thêm tài liệu mới)
- 17 nguồn, 6,127 chunks, ~3.1 triệu token (embedding)
- Chi phí: dưới $0.10 (embedding rẻ, $0.02/1M token)

### Ước tính 1 phiên demo (10 câu hỏi hỗn hợp)
~$0.06 (10 × $0.0059 trung bình) ≈ **1.560 VND** (tỷ giá tham khảo 26.000 VND/USD)

## Đề xuất tối ưu chi phí

1. **Giới hạn số lần lặp `search_knowledge_base` không cải thiện**
   - Mô tả: case `both_007` gọi `search_knowledge_base` lặp lại tới khi hết
     `MAX_AGENT_ITERATIONS` (5 bước) mà không dừng.
   - Tiết kiệm ước lượng: `both_007` tốn **$0.028888** (5 iterations, hit max) so với trung
     bình các case `both` còn lại **$0.009107** (không hit max) — **gấp ~3,17 lần**. Giới hạn
     2-3 lần thử ước tính tiết kiệm ~$0.02/câu cho nhóm câu hỏi rơi vào pattern này.
   - Rủi ro: thay đổi system prompt không chính xác có thể khiến agent bỏ cuộc quá sớm với
     câu hỏi hợp lệ cần nhiều bước tra cứu thật (nguồn: `DATA_SOURCES.md`, mục
     "search-until-exhausted pattern" — đây chính là lý do fix này **chưa** được áp dụng,
     để lại cho V2).
2. **Cache embedding câu hỏi lặp lại**
   - Mô tả: hiện mỗi câu hỏi luôn tính embedding mới qua `search_knowledge_base`.
   - Tiết kiệm ước lượng: không đáng kể về $ (embedding $0.02/1M token, 1 câu hỏi chỉ vài
     chục token → dưới $0.00001/câu) — lợi ích chính là **giảm latency** (bỏ round-trip gọi
     API embedding cho câu hỏi trùng lặp).
   - Rủi ro: cache phình to nếu không giới hạn kích thước/TTL; cần đảm bảo cache key không
     nhầm giữa các câu hỏi gần giống nhau nhưng khác nghĩa.
3. **Batch ingest thông minh** — dùng hash nội dung để skip chunk không đổi khi re-ingest.
   - Tiết kiệm ước lượng: nếu chỉ 1 tài liệu nhỏ (vd 1 báo cáo EVN, ~26 chunks) thay đổi
     trong tổng 6.127 chunks, tiết kiệm ước ~99,6% cost ingest cho lần cập nhật đó (~$0.0996
     trên tổng ~$0.10 hiện tại) — ước lượng theo tỷ lệ chunk, **chưa đo thật vì chưa triển khai**.
   - Rủi ro: nếu logic hash/invalidate sai, có thể bỏ sót cập nhật nội dung thật (serve
     embedding cũ dù nội dung đã đổi).

## Rủi ro chi phí cần lưu ý

1. **Corpus lớn (6,127 chunks) làm tăng token context** khi 1 tool trả về payload lớn
   (đã ghi nhận với group_metric ở M8).
2. **API key có hạn dùng, ngân sách cố định** (limit_reset: null) — theo dõi qua GET /api/v1/key.
3. **Giá OpenRouter dao động theo provider** được tự động chọn mỗi lần gọi.
4. **Pattern "search-until-exhausted"** (mục Đề xuất tối ưu #1) có thể làm chi phí trung bình
   tăng bất thường nếu tỷ lệ câu hỏi khó tăng lên trong thực tế sử dụng.
