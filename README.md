# Electricity Consumption AI Agent

AI agent phân tích dữ liệu tiêu thụ điện và tra cứu tri thức ngành điện bằng tiếng Việt,
kết hợp phân tích dữ liệu có cấu trúc (Pandas) và tra cứu tài liệu (RAG).

## Tính năng chính
- Phân tích dữ liệu CSV/Excel qua 6 công cụ Pandas (trung bình, gom nhóm, tìm đỉnh, so sánh,
  tương quan, tổng quan dataset)
- Tra cứu tri thức từ 17 tài liệu (2 báo cáo EVN, 1 văn bản chính sách VN, 14 báo cáo IEA quốc tế)
- Kết hợp cả hai (BOTH mode) — tra cứu quy định rồi tính toán áp dụng
- Trích dẫn nguồn chính xác (tên tài liệu + trang/mục)
- Upload/quản lý (thêm, xóa) tài liệu KB ngay trong lúc chạy app, không cần khởi động lại
- Ghi log chi tiết + theo dõi chi phí mỗi lượt hỏi
- Phòng chống prompt injection cơ bản

## Kiến trúc

```
User → Streamlit UI → Agent Loop (OpenRouter, qwen/qwen3.6-27b)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      6 Pandas Tools                search_knowledge_base
      + calculator                        │
              │                     ChromaDB (6,127 chunks,
         DatasetManager              17 nguồn, metadata filter
         (CSV/Excel)                 theo document_type)
```

## Công nghệ sử dụng
- Python, Pandas, Streamlit
- OpenRouter (chat: qwen/qwen3.6-27b, embedding: openai/text-embedding-3-small)
- ChromaDB (PersistentClient, lưu cục bộ tại storage/chroma/)
- PyMuPDF (PDF), hỗ trợ thêm TXT/MD qua registry mở rộng được
- Pydantic (validate tham số tool)
- pytest (93 test tự động)

## Cài đặt và chạy

```bash
git clone <URL repo>
cd electricity-consumption-ai-agent
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate         # Mac/Linux
pip install -r requirements.txt
copy .env.example .env              # điền OPENROUTER_API_KEY thật vào .env
python scripts/prepare_data.py      # tải + xử lý dataset Tetouan
python scripts/ingest_kb.py         # nạp toàn bộ tài liệu KB vào ChromaDB
streamlit run app.py
```

## Chạy bằng Docker

```bash
docker build -t electricity-agent .
docker run -p 8501:8501 --env-file .env -v ${PWD}/storage/chroma:/app/storage/chroma electricity-agent
```

## Cấu trúc thư mục

```
config/     settings.py, prompts.py (system prompt)
data/       raw/processed/sample — dataset Tetouan
kb/         raw/ (PDF/TXT gốc), sources.json (metadata 17 nguồn)
tools/      dataset_manager.py, dataframe_tools.py, calculator_tools.py
rag/        extractors.py, chunker.py, embeddings.py, vector_store.py, retriever.py,
            citations.py, kb_manager.py (upload/list/xóa KB động)
agent/      tool_schemas.py, tool_registry.py, loop.py, agent.py
services/   openrouter_client.py, logger.py, cost_tracker.py
tests/      93 test tự động (pytest)
evals/      cases.jsonl (53 câu), run_evals.py, results/
scripts/    download_data.py, prepare_data.py, ingest_kb.py
app.py      Streamlit UI
```

## Dữ liệu sử dụng
- **Structured data:** Power Consumption of Tetouan City (UCI ML Repository, id=849, CC BY 4.0)
  — benchmark kỹ thuật, KHÔNG phải dữ liệu Việt Nam
- **Knowledge base (17 nguồn, 6,127 đoạn):**
  - EVN Annual Report 2024-2025, 2022-2023
  - Chỉ thị 20/CT-TTg về tăng cường tiết kiệm điện (chính sách VN)
  - 14 báo cáo IEA: World Energy Outlook (2019-2025), Energy Efficiency (2019-2025)
- Chi tiết đầy đủ về nguồn, giấy phép, giới hạn đã biết: xem DATA_SOURCES.md

## Kết quả đánh giá

Hệ thống được đánh giá deterministic ở routing (chọn đúng nhánh Data/KB/Both), tool calling
(đúng công cụ và tham số), kết quả số (qua Pandas, không phải LLM tự tính), và answerability
(từ chối đúng khi ngoài phạm vi dữ liệu). Chưa triển khai formal Recall@K/MRR/citation
precision hay LLM-as-a-Judge cho chất lượng ngữ nghĩa câu trả lời — đây là hướng mở rộng V2
(đã tham khảo phương pháp từ dataquestio/tutorials/evaluating-llm-outputs, xem DATA_SOURCES.md).

**Bộ 53 câu hỏi, 8 category — kết quả chính thức:**

| Category | Kết quả |
|---|---|
| data_simple | 10/10 (100%) |
| group_compare | 8/8 (100%) |
| multi_step | 5/5 (100%) |
| kb_qa | 14/14 (100%) |
| both | 6/7 (85.7%) |
| error_handling | 5/5 (100%) |
| prompt_injection | 3/3 (100%) |
| parametric_override | 0/1 (0%) |
| **Tổng** | **51/53 (96.2%)** |

Chi phí: $0.310503 cho 53 câu (~$0.0059/câu). Latency trung bình: 21.4s/câu.
3 case fail còn lại đều đã điều tra rõ nguyên nhân (không phải bug), xem DATA_SOURCES.md.

## Giới hạn đã biết
- Cross-lingual retrieval gap (câu hỏi tiếng Việt match kém hơn tiếng Anh với cùng nội dung)
- Một số nội dung trích xuất từ biểu đồ/chart bị xáo trộn thứ tự
- Corpus đa domain (VN + IEA quốc tế) có thể gây source-type mismatch — giảm nhẹ bằng
  document_type metadata filter
- Đã thử nghiệm hybrid search (BM25+RRF) và cross-encoder reranker — không đưa vào production
  do không cải thiện có bằng chứng đo được (xem DATA_SOURCES.md)
- Agent đôi khi lặp tìm kiếm nhiều lần không dừng đúng lúc (search-until-exhausted pattern) —
  ghi nhận, chưa fix, xem DATA_SOURCES.md
- Ingest KB lớn: đo thực tế 8.16MB → 619 chunks, 326.731 tokens, $0.006535, 33.7s; ngoại suy
  1GB → ~77.700 chunks, ~41M tokens, ~$0.82, ~70 phút — khả thi về chi phí nhưng vượt khung
  thời gian demo live
- Refusal scoring trong eval suite dùng regex/keyword heuristic, có giới hạn cố hữu do LLM trả
  lời không ổn định về cách diễn đạt (đã chuyển từ substring sang regex để giảm false-negative,
  nhưng không loại bỏ hoàn toàn — xem DATA_SOURCES.md)
- Test fixture `tests/test_kb_manager.py` không isolate ChromaDB — mỗi lần chạy test ghi vector
  thật vào `storage/chroma/`, dọn dẹp phụ thuộc hoàn toàn vào `finally` block của từng test
- Chi tiết đầy đủ: DATA_SOURCES.md phần Known Limitations

## Testing

```bash
pytest tests/ -v          # 93 test, ~5-8 phút (một số gọi API thật)
python evals/run_evals.py # 53 câu eval end-to-end (~20-30 phút, tốn API thật)
```

## Future Work
- LLM-as-a-Judge cho faithfulness/completeness (tham khảo dataquestio/tutorials)
- Formal Recall@K, MRR, Citation Precision/Recall
- Query expansion + reranking cho cross-lingual gap (đã thử nghiệm sơ bộ)
- System prompt rule giới hạn số lần lặp search_knowledge_base không dừng
