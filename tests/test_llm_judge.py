"""Test parse JSON 3 lớp của judge_answer() — không gọi API thật (mock chat_completion)."""
from unittest.mock import MagicMock, patch

from evals.llm_judge import judge_answer


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
