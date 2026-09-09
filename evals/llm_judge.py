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


def _call_judge(system_prompt: str, user_message: str, error_context: str) -> dict | None:
    """
    Logic gọi Judge model dùng CHUNG cho judge_answer() (4 chiều, batch eval có ground
    truth) VÀ 3 hàm reference-free mới (judge_faithfulness/judge_answer_relevancy/
    judge_context_precision) — tách ra đây khi thêm 3 hàm mới để KHÔNG lặp lại logic
    gọi API + parse JSON 3 lớp ở 4 nơi. Hành vi giữ NGUYÊN 100% so với code gốc của
    judge_answer() trước khi tách (đã verify qua test cũ trong tests/test_llm_judge.py
    không đổi kết quả).

    error_context: chuỗi ngắn ghi vào log khi fail (vd "question: ...", "faithfulness
    check cho answer: ..."), để phân biệt lỗi tới từ hàm judge nào khi đọc log.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        response = chat_completion(messages, model=JUDGE_MODEL, temperature=0.0)
    except Exception as e:
        print(f"[llm_judge] Judge API call failed for {error_context}")
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
            print(f"[llm_judge] JSON parse failed for {error_context}")
            print(f"[llm_judge] Raw response: {raw[:300]}")
            return None


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
    4 chiều (faithfulness, correctness, citation_quality, completeness). Dùng cho
    BATCH EVAL có ground truth sẵn (gold_evidence/expected_answer_contains) — xem
    judge_faithfulness/judge_answer_relevancy/judge_context_precision bên dưới cho
    bản REFERENCE-FREE (không cần ground truth, dùng cho UI chấm nhanh 1 câu bất kỳ).

    Trả về dict 4 chiều nếu Judge chạy OK và JSON parse được. Trả về None (KHÔNG raise)
    nếu lời gọi API thất bại hoặc JSON không parse được — xem _call_judge() để biết chi
    tiết retry/parse 3 lớp. Mọi trường hợp fail đều được log rõ ra stdout.

    temperature=0.0 cố định — Judge cần chấm điểm NHẤT QUÁN cho cùng 1 input, không
    cần đa dạng hoá câu chữ như agent chính (agent chính không truyền temperature,
    dùng default provider — không đổi hành vi đó).
    """
    user_message = _build_user_message(
        question, agent_answer, retrieved_chunks,
        gold_evidence, acceptable_alternative_evidence, expected_answer_contains,
    )
    return _call_judge(JUDGE_SYSTEM_PROMPT, user_message, f"question: {question[:60]}...")


# =============================================================================
# 3 hàm judge REFERENCE-FREE — KHÔNG cần gold_evidence/expected_answer_contains,
# dùng cho UI "chấm chi tiết" 1 câu trả lời RAG bất kỳ ngay trong lúc chat (khác
# judge_answer() ở trên vốn thiết kế cho batch eval có ground truth sẵn).
# =============================================================================

FAITHFULNESS_SYSTEM_PROMPT = """Bạn là chuyên gia đánh giá RAG (Retrieval-Augmented Generation) cho câu trả lời tiếng Việt về ngành điện Việt Nam.

Nhiệm vụ: Đọc câu trả lời (ANSWER) của agent, TÁCH thành các claim (khẳng định) nhỏ, độc lập, có thể kiểm tra riêng lẻ — mỗi claim là 1 sự kiện/số liệu/nhận định cụ thể. Với MỖI claim, kiểm tra: claim đó có suy ra được (trực tiếp hoặc gián tiếp hợp lý) từ CÁC ĐOẠN TÀI LIỆU (CHUNKS) được cung cấp không?

QUAN TRỌNG: đây KHÔNG phải kiểm tra claim có ĐÚNG THỰC TẾ KHÁCH QUAN không (không cần biết đúng/sai ngoài đời) — CHỈ kiểm tra claim có CĂN CỨ trong chunks hay là agent tự bịa thêm ngoài phạm vi chunks.

═══════════════════════════════════════
QUY TẮC TÁCH CLAIM
═══════════════════════════════════════
- Câu chào hỏi, câu dẫn nhập không mang thông tin ("Dưới đây là...", "Tôi xin trả lời...") KHÔNG tính là claim.
- Mỗi số liệu cụ thể, mỗi sự kiện, mỗi kết luận là 1 claim riêng.
- Câu ghép nhiều ý → tách thành nhiều claim độc lập.
- Nếu ANSWER là lời từ chối trung thực (vd "không tìm thấy thông tin trong tài liệu") và không khẳng định số liệu/sự kiện cụ thể nào → trả về "claims": [] (không có gì cần kiểm tra căn cứ).

═══════════════════════════════════════
VÍ DỤ ĐỐI CHỨNG
═══════════════════════════════════════
- ANSWER: "Tỷ lệ tổn thất điện năng năm 2023 là 6.05%, giảm so với năm 2022."
  → claim 1: "Tỷ lệ tổn thất điện năng năm 2023 là 6.05%" | claim 2: "Tỷ lệ này giảm so với năm 2022"
  Nếu chunk CHỈ nói "tỷ lệ tổn thất 2023 là 6.05%" mà KHÔNG nhắc gì tới 2022 → claim 1 supported=true, claim 2 supported=false (không có căn cứ so sánh với 2022 trong chunks).
- ANSWER: "Doanh thu năm 2023 của EVN là 500 nghìn tỷ đồng." Không chunk nào nhắc tới doanh thu.
  → claim: "Doanh thu năm 2023 là 500 nghìn tỷ đồng", supported=false (bịa, không có căn cứ trong chunks).

═══════════════════════════════════════
FORMAT OUTPUT — BẮT BUỘC
═══════════════════════════════════════
CHỈ trả về JSON hợp lệ theo đúng cấu trúc dưới đây. KHÔNG thêm markdown code fence, KHÔNG thêm giải thích/lời dẫn nào khác ngoài JSON:

{
  "claims": [
    {"text": "<nội dung claim>", "supported": true/false, "reason": "<giải thích ngắn 1 câu>"}
  ]
}"""


ANSWER_RELEVANCY_SYSTEM_PROMPT = """Bạn là chuyên gia đánh giá câu trả lời tiếng Việt cho hệ thống hỏi-đáp về ngành điện Việt Nam.

Nhiệm vụ: đánh giá câu trả lời (ANSWER) có THỰC SỰ trả lời đúng trọng tâm câu hỏi (QUESTION) không — không lạc đề, không né tránh vô lý, không trả lời một câu hỏi khác na ná.

═══════════════════════════════════════
THANG ĐIỂM relevancy_score (0.0 – 1.0, bước 0.1)
═══════════════════════════════════════
- 1.0: Trả lời đúng trọng tâm câu hỏi, không lan man.
- 0.5: Trả lời được một phần, hoặc có nội dung liên quan nhưng lạc mất 1 phần trọng tâm câu hỏi.
- 0.0: Hoàn toàn lạc đề, trả lời sang chuyện khác không liên quan.

LƯU Ý ĐẶC BIỆT: từ chối trả lời VÌ LÝ DO CHÍNH ĐÁNG (câu hỏi ngoài phạm vi dữ liệu/tài liệu đang có, agent thẳng thắn nói không tìm thấy thông tin) KHÔNG bị coi là lạc đề — đó vẫn là relevancy_score CAO vì agent đã "trả lời đúng" bằng cách từ chối trung thực, đúng những gì câu hỏi cần biết (là có thông tin hay không).

═══════════════════════════════════════
VÍ DỤ ĐỐI CHỨNG
═══════════════════════════════════════
- QUESTION: "Tỷ lệ hộ dân nông thôn có điện là bao nhiêu?" ANSWER: "Tỷ lệ hộ dân nông thôn có điện là 100%, nguồn: EVN 2024-2025 trang 9." → relevancy_score=1.0.
- QUESTION: "Tỷ lệ hộ dân nông thôn có điện là bao nhiêu?" ANSWER: "EVN được thành lập năm 1995 và có trụ sở tại Hà Nội." → relevancy_score=0.0 (lạc đề hoàn toàn, không trả lời tỷ lệ được hỏi).
- QUESTION: "Doanh thu năm 2023 là bao nhiêu?" ANSWER: "Tôi không tìm thấy thông tin về doanh thu năm 2023 trong tài liệu hiện có." → relevancy_score=1.0 (từ chối đúng lý do, không lạc đề).

═══════════════════════════════════════
FORMAT OUTPUT — BẮT BUỘC
═══════════════════════════════════════
CHỈ trả về JSON hợp lệ, KHÔNG markdown fence, KHÔNG giải thích ngoài JSON:

{"relevancy_score": <0.0-1.0>, "reason": "<giải thích ngắn 1-2 câu>"}"""


CONTEXT_PRECISION_SYSTEM_PROMPT = """Bạn là chuyên gia đánh giá chất lượng retrieval cho hệ thống RAG tiếng Việt về ngành điện Việt Nam.

Nhiệm vụ: các đoạn tài liệu (chunk) dưới đây được đánh số [1], [2], [3]... theo ĐÚNG thứ tự đã retrieve (rank cao nhất trước). Với MỖI chunk, chấm verdict NHỊ PHÂN: chunk đó có LIÊN QUAN tới câu hỏi (QUESTION) không — tức là chunk có khả năng chứa thông tin giúp trả lời câu hỏi, DÙ chunk đó có chứa đáp án chính xác hay không.

QUAN TRỌNG: bạn CHỈ chấm true/false cho từng chunk theo ĐÚNG thứ tự [1][2][3]... KHÔNG tự tính điểm tổng hợp — việc tính điểm (có trọng số theo rank) do code đảm nhiệm.

═══════════════════════════════════════
QUY TẮC
═══════════════════════════════════════
- true nếu chunk bàn về ĐÚNG chủ đề câu hỏi hỏi (dù chunk có thể không chứa đáp án chính xác, hoặc là số liệu năm khác/nguồn khác cùng chủ đề).
- false nếu chunk hoàn toàn KHÔNG liên quan chủ đề câu hỏi.

═══════════════════════════════════════
VÍ DỤ ĐỐI CHỨNG
═══════════════════════════════════════
- QUESTION: "Tỷ lệ tổn thất điện năm 2023 là bao nhiêu?" Chunk nói về "tỷ lệ tổn thất điện năm 2022 là 6.5%" → true (cùng chủ đề, chỉ khác năm).
- QUESTION: "Tỷ lệ tổn thất điện năm 2023 là bao nhiêu?" Chunk nói về "chính sách trợ giá xăng dầu" → false (khác chủ đề hoàn toàn, không liên quan điện).

═══════════════════════════════════════
FORMAT OUTPUT — BẮT BUỘC
═══════════════════════════════════════
CHỈ trả về JSON hợp lệ, KHÔNG markdown fence, KHÔNG giải thích ngoài JSON. Mảng "verdicts" PHẢI có ĐÚNG số phần tử bằng số chunk được đánh số, theo ĐÚNG thứ tự [1][2][3]...:

{"verdicts": [true, false, true], "explanation": "<giải thích ngắn gọn, chunk nào relevant/không và vì sao>"}"""


def judge_faithfulness(answer: str, retrieved_chunks: list[dict]) -> dict | None:
    """
    Tách answer thành các claim nhỏ, hỏi Judge từng claim có suy ra được từ chunks
    không — KHÔNG cần ground truth, chỉ cần answer + chunks. Reference-free, dùng cho
    UI chấm nhanh (khác judge_answer() ở trên, vốn cần gold_evidence/expected_answer).

    Trả về {"claims": [...], "faithfulness_score": float} — score = tỷ lệ claim được
    support (supported=true) trên tổng số claim. Nếu answer không có claim nào (vd từ
    chối trung thực) → claims=[] và faithfulness_score=1.0 (không có gì để bịa, coi là
    "trung thực" theo đúng nghĩa đen — không phải "không đánh giá được").

    Trả về None nếu API call/parse JSON thất bại (xem _call_judge()).
    """
    chunks_section = (
        "\n".join(_format_chunk(c) for c in retrieved_chunks)
        if retrieved_chunks
        else "(Không có chunk nào được retrieve.)"
    )
    user_message = f"=== ANSWER (câu trả lời cần tách claim và kiểm tra căn cứ) ===\n{answer}\n\n=== RETRIEVED CHUNKS ===\n{chunks_section}"

    result = _call_judge(FAITHFULNESS_SYSTEM_PROMPT, user_message, f"faithfulness check, answer: {answer[:60]}...")
    if result is None:
        return None

    claims = result.get("claims", [])
    if not claims:
        result["faithfulness_score"] = 1.0
    else:
        supported_count = sum(1 for c in claims if c.get("supported") is True)
        result["faithfulness_score"] = round(supported_count / len(claims), 4)
    return result


def judge_answer_relevancy(question: str, answer: str) -> dict | None:
    """
    Đánh giá answer có thực sự trả lời đúng question không (không lạc đề, không né
    tránh vô lý) — KHÔNG cần chunks, KHÔNG cần ground truth.

    Trả về {"relevancy_score": float (0-1), "reason": str}, hoặc None nếu API call/
    parse JSON thất bại.
    """
    user_message = f"=== QUESTION ===\n{question}\n\n=== ANSWER ===\n{answer}"
    return _call_judge(ANSWER_RELEVANCY_SYSTEM_PROMPT, user_message, f"answer relevancy check, question: {question[:60]}...")


def _normalize_verdicts(parsed: dict | None, expected_len: int) -> list[bool]:
    """Verdict thiếu -> False (bảo thủ). Verdict thừa -> cắt bỏ.

    Judge đôi khi trả về danh sách "verdicts" sai độ dài (thiếu 1 phần tử, hoặc lặp
    thừa) dù prompt đã yêu cầu khớp chính xác — KHÔNG raise, tự chuẩn hoá về đúng
    expected_len để _average_precision() luôn chạy được trên list cùng độ dài với
    retrieved_chunks. Verdict thiếu mặc định False (bảo thủ: coi chunk chưa được
    chấm là "không relevant" thay vì lạc quan giả định True)."""
    verdicts = parsed.get("verdicts") if parsed else None
    if not isinstance(verdicts, list):
        return [False] * expected_len
    normalized = [bool(v) for v in verdicts[:expected_len]]
    normalized += [False] * (expected_len - len(normalized))
    return normalized


def _average_precision(verdicts: list[bool]) -> float:
    """AP@k: thưởng chunk relevant đứng rank cao hơn (đầu danh sách), đúng công thức
    Average Precision chuẩn (giống ContextPrecisionScorer trong RAGAS/production-grade-rag)
    — KHÔNG phải tỷ lệ đơn giản relevant_count/total, vì tỷ lệ đơn giản không phân biệt
    được "chunk relevant nằm ở rank 1" (retrieval tốt) với "chunk relevant nằm ở rank
    cuối" (retrieval kém, chỉ tình cờ có 1 chunk đúng lẫn trong đống chunk sai)."""
    relevant_count = 0
    precision_sum = 0.0
    for i, is_relevant in enumerate(verdicts, start=1):
        if is_relevant:
            relevant_count += 1
            precision_sum += relevant_count / i
    if relevant_count == 0:
        return 0.0
    return round(precision_sum / relevant_count, 4)


def judge_context_precision(question: str, retrieved_chunks: list[dict]) -> dict | None:
    """
    Judge chấm verdict NHỊ PHÂN (relevant true/false) cho TỪNG chunk theo đúng thứ tự
    rank đã retrieve — KHÔNG để LLM tự tính điểm tổng hợp (LLM làm toán không đáng tin
    cậy bằng code). CODE tự tính context_precision_score bằng Average Precision (AP@k),
    đúng pattern ContextPrecisionScorer (RAGAS / Ashok007-cmd/production-grade-rag) —
    thưởng chunk relevant đứng rank cao hơn, phạt chunk relevant lọt xuống cuối.

    KHÔNG cần ground truth, KHÔNG cần answer.

    Trả về {"verdicts": [bool,...], "chunks": [{"source", "page", "relevant"}, ...],
    "context_precision_score": float (0-1), "explanation": str}. "chunks" được CODE
    tự ghép lại từ retrieved_chunks gốc (source/page lấy từ input, không phụ thuộc
    LLM echo lại đúng metadata) — chỉ "relevant" là verdict LLM chấm.

    Nếu retrieved_chunks rỗng → trả về context_precision_score=None NGAY, KHÔNG gọi
    API (không có gì để chấm — None nghĩa là "không đánh giá được", khác 0.0 nghĩa là
    "toàn bộ không liên quan").

    Trả về None nếu API call/parse JSON thất bại (xem _call_judge()).
    """
    if not retrieved_chunks:
        return {"verdicts": [], "chunks": [], "context_precision_score": None, "explanation": ""}

    chunks_section = "\n".join(
        f"[{i}] {_format_chunk(c)}" for i, c in enumerate(retrieved_chunks, start=1)
    )
    user_message = f"=== QUESTION ===\n{question}\n\n=== RETRIEVED CHUNKS (chấm verdict true/false theo đúng số thứ tự) ===\n{chunks_section}"

    result = _call_judge(CONTEXT_PRECISION_SYSTEM_PROMPT, user_message, f"context precision check, question: {question[:60]}...")
    if result is None:
        return None

    verdicts = _normalize_verdicts(result, len(retrieved_chunks))
    chunks_detail = [
        {"source": c.get("source"), "page": c.get("page"), "relevant": v}
        for c, v in zip(retrieved_chunks, verdicts)
    ]
    return {
        "verdicts": verdicts,
        "chunks": chunks_detail,
        "context_precision_score": _average_precision(verdicts),
        "explanation": result.get("explanation", ""),
    }


# Nguong heuristic KHOI DIEM cho diagnose_rag_quality() — CHUA duoc validate rong tren
# nhieu case that (giong cach SCORE_THRESHOLD/THRESHOLDS trong evals/run_evals.py da
# lam: dat 1 gia tri hop ly ban dau, do tren du lieu that roi moi tinh chinh neu can).
# KHONG tu tin tuyet doi vao 2 con so nay.
CONTEXT_PRECISION_ISSUE_THRESHOLD = 0.5
FAITHFULNESS_ISSUE_THRESHOLD = 0.8


def diagnose_rag_quality(question: str, answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Gọi cả 3 hàm judge reference-free (judge_faithfulness, judge_answer_relevancy,
    judge_context_precision), tổng hợp thành 1 kết luận verdict — "lỗi nằm ở Retrieval
    hay Generation":

    - context_precision THẤP (<0.5) + faithfulness CAO (>=0.8): RETRIEVAL_ISSUE
      (chunks lấy về không liên quan/xếp hạng kém, nhưng answer vẫn trung thực với
      những gì lấy được — lỗi nằm ở bước tìm kiếm, không phải bước tổng hợp câu trả lời).
    - context_precision CAO (>=0.5) + faithfulness THẤP (<0.8): GENERATION_ISSUE
      (chunks liên quan tốt, nhưng answer không bám sát/bịa thêm — lỗi nằm ở bước
      tổng hợp câu trả lời, không phải bước tìm kiếm).
    - CẢ HAI thấp: BOTH_ISSUES.
    - CẢ HAI đạt ngưỡng: OK.
    - Nếu bất kỳ lời gọi Judge nào fail (trả None) hoặc context_precision_score=None
      (không có chunk nào để chấm): verdict="JUDGE_ERROR" hoặc "NO_CONTEXT" tương ứng
      — KHÔNG cố suy luận verdict từ dữ liệu thiếu.

    Ngưỡng 0.5/0.8 là ĐIỂM KHỞI ĐẦU heuristic — xem CONTEXT_PRECISION_ISSUE_THRESHOLD/
    FAITHFULNESS_ISSUE_THRESHOLD ở trên, có thể cần điều chỉnh sau khi thấy dữ liệu
    thật, KHÔNG coi là chính xác tuyệt đối.
    """
    faithfulness = judge_faithfulness(answer, retrieved_chunks)
    relevancy = judge_answer_relevancy(question, answer)
    context_prec = judge_context_precision(question, retrieved_chunks)

    if faithfulness is None or relevancy is None or context_prec is None:
        return {
            "verdict": "JUDGE_ERROR",
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": context_prec,
        }

    context_score = context_prec["context_precision_score"]
    if context_score is None:
        return {
            "verdict": "NO_CONTEXT",
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": context_prec,
        }

    faithfulness_score = faithfulness["faithfulness_score"]
    context_low = context_score < CONTEXT_PRECISION_ISSUE_THRESHOLD
    faithfulness_low = faithfulness_score < FAITHFULNESS_ISSUE_THRESHOLD

    if context_low and not faithfulness_low:
        verdict = "RETRIEVAL_ISSUE"
    elif not context_low and faithfulness_low:
        verdict = "GENERATION_ISSUE"
    elif context_low and faithfulness_low:
        verdict = "BOTH_ISSUES"
    else:
        verdict = "OK"

    return {
        "verdict": verdict,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_precision": context_prec,
    }
