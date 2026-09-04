"""
LLM-as-a-Judge — Phase 1 smoke test. Đánh giá chất lượng câu trả lời của agent
bằng 1 model KHÁC (không phải qwen3.6-27b agent chính dùng) theo 4 chiều.

KHÔNG tích hợp vào evals/run_evals.py — module độc lập, dễ rollback nếu Judge
không hoạt động đúng như kỳ vọng ở Phase 1.
"""
import json
import re

from services.openrouter_client import chat_completion

# Model Judge — verified tồn tại thật qua GET /api/v1/models (giá $0.75/M input,
# $3.75/M output, context 1.05M), khác họ hoàn toàn với qwen3.6-27b (agent chính)
# để tránh bias tự chấm cùng nhà.
JUDGE_MODEL = "google/gemini-3.8-flash"


JUDGE_SYSTEM_PROMPT = """Bạn là chuyên gia đánh giá chất lượng câu trả lời của một AI agent trả lời câu hỏi tiếng Việt về ngành điện Việt Nam (dựa trên báo cáo thường niên EVN, văn bản chính sách tiết kiệm điện, và báo cáo năng lượng quốc tế IEA). Agent này dùng kỹ thuật RAG (Retrieval-Augmented Generation): nó tìm các đoạn tài liệu (chunks) liên quan, rồi tổng hợp câu trả lời dựa trên các chunks đó.

Nhiệm vụ của bạn: chấm điểm câu trả lời của agent theo ĐÚNG 4 tiêu chí dưới đây, mỗi tiêu chí thang điểm 1-5.

═══════════════════════════════════════
TIÊU CHÍ 1 — faithfulness (độ trung thực với nguồn)
═══════════════════════════════════════
Kiểm tra: câu trả lời có KHẲNG ĐỊNH số liệu hoặc sự kiện nào KHÔNG xuất hiện trong các chunks retrieve được không? Đây là kiểm tra "agent có bịa thêm nội dung ngoài chunks hay không", KHÔNG PHẢI kiểm tra câu trả lời có đúng thực tế khách quan hay không (đó là tiêu chí correctness riêng).

Thang điểm:
- 1: Câu trả lời khẳng định số liệu/sự kiện HOÀN TOÀN không xuất hiện trong bất kỳ chunk nào (bịa đặt nghiêm trọng, không có căn cứ).
- 2: Phần lớn nội dung không truy được về chunks, chỉ có 1-2 chi tiết nhỏ khớp.
- 3: Có sự pha trộn — một số claim khớp chunks rõ ràng, một số claim không xác định được là có trong chunks hay không (mơ hồ, không chắc chắn).
- 4: Hầu hết claim khớp chunks, có 1 chi tiết nhỏ là suy luận/diễn giải hợp lý từ chunks (không trích nguyên văn) nhưng không phải bịa mới.
- 5: MỌI số liệu và sự kiện trong câu trả lời đều truy được nguồn gốc trực tiếp từ ít nhất 1 chunk retrieve được, không có nội dung bịa thêm.

Ví dụ đối chứng:
- Điểm 5: Chunk ghi "tỷ lệ tổn thất điện năng năm 2023 là 6.05%". Answer: "Tỷ lệ tổn thất điện năng của EVN năm 2023 là 6.05%." → mọi số liệu đều khớp chunk.
- Điểm 1: Không chunk nào nhắc tới doanh thu. Answer: "Doanh thu năm 2023 của EVN là 500 nghìn tỷ đồng." → số liệu bịa, không có căn cứ trong chunks.

═══════════════════════════════════════
TIÊU CHÍ 2 — correctness (đúng sự thật)
═══════════════════════════════════════
Kiểm tra: câu trả lời có ĐÚNG so với gold_evidence / expected_answer_contains (đáp án chuẩn đã biết trước) không?

Thang điểm:
- 1: Kết luận/số liệu chính SAI HẲN so với đáp án chuẩn (ngược nghĩa hoặc số liệu hoàn toàn khác).
- 2: Có chứa đúng một phần rất nhỏ nhưng kết luận chính vẫn sai.
- 3: Đúng đại ý nhưng số liệu cụ thể bị lệch (vd làm tròn sai, nhầm đơn vị, nhầm năm).
- 4: Đúng gần như hoàn toàn, chỉ khác biệt rất nhỏ về cách diễn đạt, không ảnh hưởng ý nghĩa.
- 5: Khớp chính xác với đáp án chuẩn, không có sai lệch nào.

Ví dụ đối chứng:
- Điểm 5: expected_answer_contains chứa "99.74". Answer chứa "99,74%" → khớp đúng (dấu phẩy/chấm thập phân không tính là sai).
- Điểm 1: expected_answer_contains chứa "99.74". Answer nói "khoảng 85%" → sai hẳn về số liệu.

═══════════════════════════════════════
TIÊU CHÍ 3 — citation_quality (chất lượng trích dẫn)
═══════════════════════════════════════
Kiểm tra: trích dẫn trong câu trả lời có đúng nguồn tài liệu + đúng trang/mục + trình bày rõ ràng không?

Thang điểm:
- 1: Không có trích dẫn nào, HOẶC trích dẫn sai hoàn toàn nguồn/trang so với gold_evidence và mọi acceptable_alternative_evidence.
- 2: Có trích dẫn nhưng thiếu 1 phần quan trọng (chỉ có tên nguồn không có trang/mục, hoặc ngược lại) và không khớp đáp án chuẩn.
- 3: Trích đúng tên nguồn nhưng số trang/mục sai hoặc thiếu.
- 4: Trích đúng nguồn + trang/mục khớp gold_evidence hoặc 1 acceptable_alternative_evidence, nhưng cách trình bày hơi khó đọc/không nhất quán.
- 5: Trích dẫn đầy đủ, chính xác (đúng nguồn + đúng trang/mục, khớp gold_evidence HOẶC 1 acceptable_alternative_evidence), trình bày rõ ràng dễ đối chiếu.

Ví dụ đối chứng:
- Điểm 5: gold_evidence là {"source_fragment": "2024-2025", "page": 9}. Answer ghi "Nguồn: EVN Annual Report 2024-2025, trang 9" → khớp chính xác.
- Điểm 1: Answer không trích dẫn nguồn nào, dù có chunks retrieve được rõ ràng.

═══════════════════════════════════════
TIÊU CHÍ 4 — completeness (đầy đủ ý)
═══════════════════════════════════════
Kiểm tra: câu trả lời có đủ ý mà câu hỏi cần không — không thiếu ý quan trọng, không lan man ra ngoài phạm vi câu hỏi?

Thang điểm:
- 1: Không trả lời được ý chính của câu hỏi, hoặc lạc đề hoàn toàn.
- 2: Chỉ trả lời được một phần nhỏ, thiếu nhiều ý quan trọng.
- 3: Trả lời được ý chính nhưng thiếu một số chi tiết phụ đáng lẽ nên có.
- 4: Đầy đủ ý chính, có thể thiếu 1 chi tiết rất nhỏ hoặc hơi lan man thêm nội dung không cần thiết.
- 5: Trả lời đầy đủ, chính xác đúng những gì câu hỏi cần, không thiếu không lan man.

Ví dụ đối chứng:
- Điểm 5: Câu hỏi "Tỷ lệ hộ dân nông thôn có điện là bao nhiêu?". Answer trả lời đúng con số, không thêm nội dung không liên quan.
- Điểm 1: Câu hỏi hỏi về tỷ lệ điện khí hóa, Answer chỉ nói về lịch sử thành lập EVN, không trả lời được số liệu cần.

═══════════════════════════════════════
QUY TẮC ĐẶC BIỆT — acceptable_alternative_evidence
═══════════════════════════════════════
Nếu câu trả lời trích dẫn một chunk KHÁC với gold_evidence chính, nhưng chunk đó nằm trong danh sách acceptable_alternative_evidence được cung cấp — đây là trích dẫn HỢP LỆ. KHÔNG được chấm thấp citation_quality hoặc correctness chỉ vì answer dùng alternative thay vì gold chính. Cả 2 đều coi là đúng.

═══════════════════════════════════════
FORMAT OUTPUT — BẮT BUỘC
═══════════════════════════════════════
CHỈ trả về JSON hợp lệ theo đúng cấu trúc dưới đây. KHÔNG thêm markdown code fence (```json ... ```), KHÔNG thêm bất kỳ giải thích, lời dẫn, hay văn bản nào khác ngoài JSON:

{
  "faithfulness": {"score": <1-5>, "notes": "<giải thích ngắn 1-2 câu>"},
  "correctness": {"score": <1-5>, "notes": "<giải thích ngắn 1-2 câu>"},
  "citation_quality": {"score": <1-5>, "notes": "<giải thích ngắn 1-2 câu>"},
  "completeness": {"score": <1-5>, "notes": "<giải thích ngắn 1-2 câu>"}
}"""


def _extract_json_object(text: str) -> dict | None:
    """Fallback cuối (lớp 3): đếm độ sâu ngoặc {} để tìm khối JSON cân bằng đầu tiên
    trong text — bắt được trường hợp Judge thêm giải thích/lời dẫn trước hoặc sau
    JSON dù system prompt đã yêu cầu chỉ trả JSON thuần (dòng 95 ở trên)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _format_chunk(chunk: dict) -> str:
    # Full text — Gemini Flash context 1.05M dư sức, không cần cắt. Cắt preview 300
    # ký tự trước đây gây bug thật: gold chunk EVN dài ~3000 ký tự, số liệu quan
    # trọng (vd "99.74%") nằm ở vị trí ~1800, ngoài 300 ký tự đầu → Judge chấm
    # faithfulness=1 sai vì không "nhìn thấy" số agent đã trích đúng. Xem báo cáo
    # Phase 1 Bước 3 (lần chạy đầu, trước fix).
    chunk_text = chunk.get("text", "")
    return f"- Nguồn: {chunk.get('source')} | Trang/mục: {chunk.get('page')} | Nội dung: \"{chunk_text}\""


def _build_user_message(
    question: str,
    agent_answer: str,
    retrieved_chunks: list[dict],
    gold_evidence: dict | None,
    acceptable_alternative_evidence: list[dict] | None,
    expected_answer_contains: list[str] | None,
) -> str:
    parts = []

    parts.append("=== QUESTION ===")
    parts.append(question)

    parts.append("\n=== ANSWER (câu trả lời của agent cần chấm điểm) ===")
    parts.append(agent_answer)

    parts.append("\n=== RETRIEVED CHUNKS (nội dung agent thực sự lấy được) ===")
    if retrieved_chunks:
        parts.append("\n".join(_format_chunk(c) for c in retrieved_chunks))
    else:
        parts.append("(Không có chunk nào được retrieve — câu hỏi không cần tra cứu tài liệu.)")

    parts.append("\n=== GOLD EVIDENCE (đáp án chuẩn về nguồn trích dẫn) ===")
    if gold_evidence:
        parts.append(
            f"- source_fragment: \"{gold_evidence.get('source_fragment')}\" | page: {gold_evidence.get('page')}"
        )
    else:
        parts.append("(Không có gold_evidence cho câu này.)")

    parts.append("\n=== ACCEPTABLE ALTERNATIVE EVIDENCE (các nguồn khác cũng được coi là đúng) ===")
    if acceptable_alternative_evidence:
        parts.append(
            "\n".join(
                f"- source_fragment: \"{alt.get('source_fragment')}\" | page: {alt.get('page')}"
                for alt in acceptable_alternative_evidence
            )
        )
    else:
        parts.append("(Không có alternative evidence nào — chỉ gold_evidence là hợp lệ.)")

    parts.append("\n=== EXPECTED ANSWER CONTAINS (câu trả lời đúng phải chứa 1 trong các chuỗi sau) ===")
    if expected_answer_contains:
        parts.append(", ".join(f'"{s}"' for s in expected_answer_contains))
    else:
        parts.append("(Không có ràng buộc nội dung cụ thể cho câu này.)")

    return "\n".join(parts)


def judge_answer(
    question: str,
    agent_answer: str,
    retrieved_chunks: list[dict],
    gold_evidence: dict | None,
    acceptable_alternative_evidence: list[dict] | None,
    expected_answer_contains: list[str] | None,
) -> dict | None:
    """
    Gọi Judge model (JUDGE_MODEL, khác agent chính) để chấm điểm 1 câu trả lời theo
    4 chiều (faithfulness, correctness, citation_quality, completeness).

    Trả về dict 4 chiều nếu Judge chạy OK và JSON parse được.
    Trả về None (KHÔNG raise) nếu:
    - Lời gọi API tới Judge model thất bại sau khi chat_completion() đã tự retry
      (chat_completion raise RuntimeError sau MAX_RETRIES lần thử)
    - Judge trả về nội dung không parse được thành JSON, kể cả sau khi đã thử cả 3
      lớp: json.loads() trực tiếp, strip markdown code fence, và brace-depth walk
      (_extract_json_object, bắt trường hợp Judge lỡ thêm giải thích quanh JSON)
    Mọi trường hợp fail đều được log rõ ra stdout (không nuốt lỗi im lặng).

    temperature=0.0 cố định — Judge cần chấm điểm NHẤT QUÁN cho cùng 1 input, không
    cần đa dạng hoá câu chữ như agent chính (agent chính không truyền temperature,
    dùng default provider — không đổi hành vi đó).
    """
    user_message = _build_user_message(
        question, agent_answer, retrieved_chunks,
        gold_evidence, acceptable_alternative_evidence, expected_answer_contains,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = chat_completion(messages, model=JUDGE_MODEL, temperature=0.0)
    except Exception as e:
        print(f"[llm_judge] Judge API call failed for question: {question[:60]}...")
        print(f"[llm_judge] Error: {e}")
        return None

    raw = response.choices[0].message.content or ""

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            fallback = _extract_json_object(raw)
            if fallback is not None:
                return fallback
            print(f"[llm_judge] JSON parse failed for question: {question[:60]}...")
            print(f"[llm_judge] Raw response: {raw[:300]}")
            return None
