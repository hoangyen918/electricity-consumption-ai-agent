"""Test parse JSON 3 lớp của judge_answer() — không gọi API thật (mock chat_completion)."""
import json
from unittest.mock import MagicMock, patch

from evals.llm_judge import (
    judge_answer,
    judge_faithfulness,
    judge_answer_relevancy,
    judge_context_precision,
    diagnose_rag_quality,
    _normalize_verdicts,
    _average_precision,
)


def _fake_response(content: str):
    """Giả lập response object của OpenAI SDK — chỉ cần .choices[0].message.content
    vì đó là field duy nhất judge_answer() đọc từ response."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


def test_judge_answer_parses_json_wrapped_in_markdown_fence():
    """Lớp 2: Judge trả JSON bọc trong ```json ... ``` dù system prompt đã yêu cầu
    không thêm fence — vẫn phải parse đúng."""
    fenced_content = (
        "```json\n"
        '{"faithfulness": {"score": 5, "notes": "ok"}, '
        '"correctness": {"score": 4, "notes": "ok"}, '
        '"citation_quality": {"score": 3, "notes": "ok"}, '
        '"completeness": {"score": 5, "notes": "ok"}}\n'
        "```"
    )
    with patch("evals.llm_judge.chat_completion", return_value=_fake_response(fenced_content)):
        result = judge_answer(
            question="q", agent_answer="a", retrieved_chunks=[],
            gold_evidence=None, acceptable_alternative_evidence=None,
            expected_answer_contains=None,
        )
    assert result is not None
    assert result["faithfulness"]["score"] == 5
    assert result["correctness"]["score"] == 4
    assert result["citation_quality"]["score"] == 3
    assert result["completeness"]["score"] == 5


def test_judge_answer_parses_json_surrounded_by_explanation_text():
    """Lớp 3 (mới, brace-depth walk): Judge thêm câu giải thích trước/sau JSON —
    lớp 1 (json.loads trực tiếp) và lớp 2 (strip markdown fence) đều fail vì có
    text xen lẫn, chỉ lớp 3 mới bắt được."""
    messy_content = (
        "Đây là kết quả đánh giá của tôi: "
        '{"faithfulness": {"score": 2, "notes": "thiếu căn cứ"}, '
        '"correctness": {"score": 1, "notes": "sai số liệu"}, '
        '"citation_quality": {"score": 1, "notes": "không trích dẫn"}, '
        '"completeness": {"score": 3, "notes": "thiếu ý"}} '
        "Hy vọng hữu ích."
    )
    with patch("evals.llm_judge.chat_completion", return_value=_fake_response(messy_content)):
        result = judge_answer(
            question="q", agent_answer="a", retrieved_chunks=[],
            gold_evidence=None, acceptable_alternative_evidence=None,
            expected_answer_contains=None,
        )
    assert result is not None
    assert result["faithfulness"]["score"] == 2
    assert result["correctness"]["score"] == 1
    assert result["citation_quality"]["score"] == 1
    assert result["completeness"]["score"] == 3


# =============================== judge_faithfulness ===============================

def test_judge_faithfulness_computes_score_from_supported_claims():
    content = json.dumps({
        "claims": [
            {"text": "claim A", "supported": True, "reason": "khớp chunk"},
            {"text": "claim B", "supported": False, "reason": "không có căn cứ"},
            {"text": "claim C", "supported": True, "reason": "khớp chunk"},
        ]
    })
    with patch("evals.llm_judge.chat_completion", return_value=_fake_response(content)):
        result = judge_faithfulness(answer="a", retrieved_chunks=[{"source": "s", "page": 1, "text": "t"}])
    assert result is not None
    assert len(result["claims"]) == 3
    assert result["faithfulness_score"] == round(2 / 3, 4)


def test_judge_faithfulness_empty_claims_scores_1_0():
    """Answer khong co claim nao (vd tu choi trung thuc) -> khong co gi de bia,
    faithfulness_score=1.0 theo dung nghia den, khong phai 'khong danh gia duoc'."""
    content = json.dumps({"claims": []})
    with patch("evals.llm_judge.chat_completion", return_value=_fake_response(content)):
        result = judge_faithfulness(answer="Tôi không tìm thấy thông tin.", retrieved_chunks=[])
    assert result is not None
    assert result["faithfulness_score"] == 1.0


# =============================== judge_answer_relevancy ===============================

def test_judge_answer_relevancy_parses_score():
    content = json.dumps({"relevancy_score": 0.8, "reason": "tra loi dung trong tam"})
    with patch("evals.llm_judge.chat_completion", return_value=_fake_response(content)):
        result = judge_answer_relevancy(question="q", answer="a")
    assert result is not None
    assert result["relevancy_score"] == 0.8


# =============================== _average_precision (AP@k formula) ===============================

def test_average_precision_matches_hand_calculation_mixed():
    """[True, False, True]: rank1 relevant->precision 1/1=1.0, rank3 relevant->precision
    2/3=0.6667 -> AP=(1.0+0.6667)/2=0.8333."""
    assert _average_precision([True, False, True]) == 0.8333


def test_average_precision_all_false_is_zero():
    assert _average_precision([False, False]) == 0.0


def test_average_precision_empty_is_zero():
    assert _average_precision([]) == 0.0


def test_average_precision_all_true_is_one():
    assert _average_precision([True, True, True]) == 1.0


# =============================== _normalize_verdicts ===============================

def test_normalize_verdicts_missing_padded_with_false():
    """Judge tra ve it verdict hon so chunk thuc te -> phan con thieu mac dinh False
    (bao thu, khong doan lac quan)."""
    result = _normalize_verdicts({"verdicts": [True, False]}, expected_len=4)
    assert result == [True, False, False, False]


def test_normalize_verdicts_extra_truncated():
    """Judge tra ve nhieu verdict hon so chunk thuc te -> cat bo phan thua."""
    result = _normalize_verdicts({"verdicts": [True, False, True, True, False]}, expected_len=3)
    assert result == [True, False, True]


def test_normalize_verdicts_none_or_missing_key_returns_all_false():
    assert _normalize_verdicts(None, expected_len=2) == [False, False]
    assert _normalize_verdicts({}, expected_len=2) == [False, False]
    assert _normalize_verdicts({"verdicts": "not a list"}, expected_len=2) == [False, False]


# =============================== judge_context_precision ===============================

def test_judge_context_precision_computes_ap_from_verdicts():
    content = json.dumps({"verdicts": [True, False], "explanation": "chunk 1 lien quan, chunk 2 khong"})
    chunks_in = [{"source": "s1", "page": 1, "text": "t1"}, {"source": "s2", "page": 2, "text": "t2"}]
    with patch("evals.llm_judge.chat_completion", return_value=_fake_response(content)):
        result = judge_context_precision(question="q", retrieved_chunks=chunks_in)
    assert result is not None
    assert result["context_precision_score"] == 1.0  # chunk relevant duy nhat o rank 1
    assert result["chunks"] == [
        {"source": "s1", "page": 1, "relevant": True},
        {"source": "s2", "page": 2, "relevant": False},
    ]


def test_judge_context_precision_handles_short_verdicts_list_from_llm():
    """LLM tra ve thieu verdict (vd 3 chunk nhung chi tra 2 verdict) -> khong crash,
    verdict thieu tinh la False qua _normalize_verdicts."""
    content = json.dumps({"verdicts": [True], "explanation": "..."})
    chunks_in = [
        {"source": "s1", "page": 1, "text": "t1"},
        {"source": "s2", "page": 2, "text": "t2"},
        {"source": "s3", "page": 3, "text": "t3"},
    ]
    with patch("evals.llm_judge.chat_completion", return_value=_fake_response(content)):
        result = judge_context_precision(question="q", retrieved_chunks=chunks_in)
    assert result is not None
    assert result["verdicts"] == [True, False, False]


def test_judge_context_precision_empty_chunks_returns_none_score_without_api_call():
    """Khong co chunk nao -> tra ve NGAY, KHONG goi API (khong co gi de cham)."""
    with patch("evals.llm_judge.chat_completion") as mock_call:
        result = judge_context_precision(question="q", retrieved_chunks=[])
    assert result == {"verdicts": [], "chunks": [], "context_precision_score": None, "explanation": ""}
    mock_call.assert_not_called()


# =============================== diagnose_rag_quality — logic cheo bang ===============================

def _mock_judges(faithfulness_score, context_score, relevancy_score=1.0):
    """Mock ca 3 ham judge_* de test THUAN logic cheo bang trong diagnose_rag_quality(),
    khong phu thuoc noi dung JSON that."""
    return (
        patch("evals.llm_judge.judge_faithfulness", return_value={"claims": [], "faithfulness_score": faithfulness_score}),
        patch("evals.llm_judge.judge_answer_relevancy", return_value={"relevancy_score": relevancy_score, "reason": "r"}),
        patch("evals.llm_judge.judge_context_precision", return_value={"verdicts": [], "chunks": [], "context_precision_score": context_score, "explanation": ""}),
    )


def test_diagnose_rag_quality_verdict_retrieval_issue():
    """context thap, faithfulness cao -> RETRIEVAL_ISSUE (chunks khong lien quan
    nhung answer van trung thuc voi nhung gi lay duoc)."""
    p1, p2, p3 = _mock_judges(faithfulness_score=0.9, context_score=0.2)
    with p1, p2, p3:
        result = diagnose_rag_quality(question="q", answer="a", retrieved_chunks=[{"source": "s", "page": 1, "text": "t"}])
    assert result["verdict"] == "RETRIEVAL_ISSUE"


def test_diagnose_rag_quality_verdict_generation_issue():
    """context cao, faithfulness thap -> GENERATION_ISSUE (chunks tot nhung answer bia them)."""
    p1, p2, p3 = _mock_judges(faithfulness_score=0.3, context_score=0.9)
    with p1, p2, p3:
        result = diagnose_rag_quality(question="q", answer="a", retrieved_chunks=[{"source": "s", "page": 1, "text": "t"}])
    assert result["verdict"] == "GENERATION_ISSUE"


def test_diagnose_rag_quality_verdict_both_issues():
    p1, p2, p3 = _mock_judges(faithfulness_score=0.2, context_score=0.1)
    with p1, p2, p3:
        result = diagnose_rag_quality(question="q", answer="a", retrieved_chunks=[{"source": "s", "page": 1, "text": "t"}])
    assert result["verdict"] == "BOTH_ISSUES"


def test_diagnose_rag_quality_verdict_ok():
    p1, p2, p3 = _mock_judges(faithfulness_score=1.0, context_score=1.0)
    with p1, p2, p3:
        result = diagnose_rag_quality(question="q", answer="a", retrieved_chunks=[{"source": "s", "page": 1, "text": "t"}])
    assert result["verdict"] == "OK"


def test_diagnose_rag_quality_boundary_values_exactly_at_threshold_count_as_ok():
    """Bien: dung 0.5/0.8 (khong phai < ma la ==) phai duoc tinh la DAT nguong, khong
    phai 'thap' (dung < nghiem ngat trong logic, khong phai <=)."""
    p1, p2, p3 = _mock_judges(faithfulness_score=0.8, context_score=0.5)
    with p1, p2, p3:
        result = diagnose_rag_quality(question="q", answer="a", retrieved_chunks=[{"source": "s", "page": 1, "text": "t"}])
    assert result["verdict"] == "OK"


def test_diagnose_rag_quality_no_context_when_no_chunks():
    with patch("evals.llm_judge.judge_faithfulness", return_value={"claims": [], "faithfulness_score": 1.0}), \
         patch("evals.llm_judge.judge_answer_relevancy", return_value={"relevancy_score": 1.0, "reason": "r"}), \
         patch("evals.llm_judge.judge_context_precision", return_value={"verdicts": [], "chunks": [], "context_precision_score": None, "explanation": ""}):
        result = diagnose_rag_quality(question="q", answer="a", retrieved_chunks=[])
    assert result["verdict"] == "NO_CONTEXT"


def test_diagnose_rag_quality_judge_error_when_any_judge_fails():
    with patch("evals.llm_judge.judge_faithfulness", return_value=None), \
         patch("evals.llm_judge.judge_answer_relevancy", return_value={"relevancy_score": 1.0, "reason": "r"}), \
         patch("evals.llm_judge.judge_context_precision", return_value={"verdicts": [], "chunks": [], "context_precision_score": 1.0, "explanation": ""}):
        result = diagnose_rag_quality(question="q", answer="a", retrieved_chunks=[{"source": "s", "page": 1, "text": "t"}])
    assert result["verdict"] == "JUDGE_ERROR"
