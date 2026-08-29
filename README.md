# Electricity Consumption AI Agent

AI agent phân tích dữ liệu tiêu thụ điện và tra cứu tri thức ngành điện bằng tiếng Việt,
kết hợp phân tích dữ liệu có cấu trúc (Pandas) và tra cứu tài liệu (RAG).

## Tính năng chính

- Phân tích dữ liệu CSV/Excel qua 6 công cụ Pandas (trung bình, gom nhóm, tìm đỉnh, so sánh,
  tương quan, tổng quan dataset)
- Tra cứu tri thức từ 17 tài liệu (báo cáo EVN, văn bản chính sách, báo cáo IEA quốc tế)
- Kết hợp cả hai (BOTH mode) — tra cứu quy định rồi tính toán áp dụng
- Trích dẫn nguồn chính xác (tên tài liệu + trang/mục)
- Upload/quản lý (thêm, xóa) tài liệu KB ngay trong lúc chạy app, không cần khởi động lại
- Ghi log chi tiết + theo dõi chi phí mỗi lượt hỏi
- Phòng chống prompt injection cơ bản

## Kiến trúc

```
User
  │
  ▼
Streamlit UI (app.py)
  │  session_state: DatasetManager, Agent, chat_history
  ▼
Agent Loop (agent/loop.py)
  │  gửi messages + 8 tool schemas
  ▼
OpenRouter — qwen/qwen3.6-27b  ◄──────────────┐
  │  trả lời text HOẶC tool_calls              │ tool result
  ▼                                            │
Tool Registry (agent/tool_registry.py)         │
  │  validate argument bằng Pydantic           │
  ├──► 6 Pandas tools ──► DatasetManager ──► CSV/Excel dataset
  ├──► calculate ──► phép tính whitelist (không eval())
  └──► search_knowledge_base ─────────────────┘
           │
           ▼
      ChromaDB (embedding: openai/text-embedding-3-small)
      17 nguồn tài liệu, 6,127 chunks
```

## Công nghệ sử dụng

- Python, Pandas, Streamlit
- OpenRouter (model chat: qwen/qwen3.6-27b, model embedding: openai/text-embedding-3-small)
- ChromaDB (PersistentClient, lưu cục bộ)
- PyMuPDF (trích xuất PDF), hỗ trợ thêm TXT/MD
- Pydantic (validate tham số tool)
- pytest (89 test tự động)

## Cài đặt và chạy (mục tiêu: dưới 10 phút từ lúc clone)

```bash
git clone <URL repo>
cd electricity-consumption-ai-agent
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate         # Mac/Linux
pip install -r requirements.txt
copy .env.example .env              # rồi điền OPENROUTER_API_KEY thật vào .env
python scripts/prepare_data.py      # tải + xử lý dataset Tetouan (nếu chưa có data/sample/)
python scripts/ingest_kb.py         # nạp toàn bộ tài liệu KB vào ChromaDB
streamlit run app.py
```

## Chạy bằng Docker

```bash
docker build -t electricity-agent .
docker run -p 8501:8501 --env-file .env electricity-agent
```

## Cấu trúc thư mục

```
config/     — settings.py (đọc .env qua pydantic-settings), prompts.py (SYSTEM_PROMPT)
data/       — raw/processed/sample dataset Tetouan City Power Consumption
kb/         — sources.json (metadata 17 nguồn), raw/ (file PDF/TXT gốc)
tools/      — DatasetManager, 6 hàm phân tích Pandas, calculator, tool_results wrapper
rag/        — extractors, chunker, embeddings, vector_store (Chroma), retriever, citations, kb_manager
agent/      — agent.py (session), loop.py (vòng lặp tool-calling), tool_schemas.py, tool_registry.py
services/   — openrouter_client.py, logger.py (JSONL trace), cost_tracker.py
tests/      — 89 test tự động (unit + integration + LLM thật)
evals/      — cases.jsonl (53 câu), run_evals.py, results/ (kết quả chi tiết)
scripts/    — download_data.py, prepare_data.py, ingest_kb.py, inspect_chroma.py, inspect_pdfs.py
storage/    — dữ liệu ChromaDB persistent (gitignored)
logs/       — agent_trace.jsonl (log mỗi lượt hỏi-đáp)
app.py      — giao diện Streamlit (chat + sidebar quản lý dataset/KB)
```

## Dữ liệu sử dụng

- **Structured data:** Power Consumption of Tetouan City (UCI ML Repository, id=849, CC BY 4.0) —
  dùng làm benchmark kỹ thuật, KHÔNG phải dữ liệu Việt Nam
- **Knowledge base** (17 nguồn, chia theo domain):
  - EVN (2): EVN Annual Report 2024-2025, EVN Annual Report 2022-2023
  - Chính sách Việt Nam (1): Chỉ thị 20/CT-TTg về tăng cường tiết kiệm điện giai đoạn 2023-2025
  - IEA quốc tế (14): World Energy Outlook 2019-2025 (7 báo cáo), Energy Efficiency 2019-2025 (7 báo cáo)
- Chi tiết đầy đủ về nguồn, giấy phép, giới hạn đã biết: xem [DATA_SOURCES.md](DATA_SOURCES.md)

## Kết quả đánh giá (evaluation)

<!-- EVAL_RESULTS_PLACEHOLDER -->

## Giới hạn đã biết

- Cross-lingual retrieval gap (câu hỏi tiếng Việt match kém hơn tiếng Anh với cùng nội dung)
- Một số nội dung trích xuất từ biểu đồ/chart bị xáo trộn (không có structure recovery)
- Corpus lớn (IEA + EVN) có thể gây source-type mismatch, giảm nhẹ bằng document_type filter
- Không dùng multi-query/reranking/hybrid search cho V1 (đã thử nghiệm hybrid search, không đưa
  vào production do không cải thiện có bằng chứng — xem DATA_SOURCES.md)
- Hai báo cáo EVN (2022-2023 và 2024-2025) công bố số liệu khác nhau cho cùng chỉ số/năm 2023 —
  agent được hướng dẫn ưu tiên báo cáo mới nhất và minh bạch nêu rõ mâu thuẫn khi phát hiện
- Chi tiết đầy đủ: xem [DATA_SOURCES.md](DATA_SOURCES.md) phần Known Limitations

## Testing

```bash
pytest tests/ -v          # 89 test, ~5-8 phút (gọi API thật ở một số test)
python evals/run_evals.py # 53 câu eval end-to-end
```
