"""Test agent/loop.py xử lý response khi message.content thiếu (None) — mock hoàn
toàn chat_completion() và execute_tool(), KHÔNG gọi API thật.

Bối cảnh: 1 endpoint đang được thử nghiệm (proxy dạng OpenAI-compatible) trả về
message.tool_calls kèm reasoning_content, nhưng KHÔNG có content khi finish_reason=
"tool_calls" — response.choices[0].message.content=None trong trường hợp này (đúng
chuẩn OpenAI: message chỉ có content khi finish_reason="stop", không có khi model
gọi tool). Test này xác nhận code hiện tại (agent/loop.py) đã xử lý AN TOÀN trường
hợp content=None hay chưa, KHÔNG giả định phải sửa gì — nếu test pass ngay không cần
sửa code, đó là kết quả hợp lệ (code đã an toàn từ trước)."""
from unittest.mock import MagicMock, patch

from agent.loop import (
    run_agent_loop,
    _tools_for_next_call,
    MAX_SEARCH_KB_CALLS,
    _looks_like_leaked_internal_tokens,
    EMPTY_CONTENT_MAX_RETRIES,
)
from agent.tool_schemas import ALL_TOOL_SCHEMAS, CALCULATE_SCHEMA
from tools.tool_results import success

# Chuỗi rác THẬT đã quan sát từ gpt-oss qua tunnel (không phải mẫu tự bịa) — xem
# báo cáo verify guard search-limit, tái hiện được cả both_001 (search->calculate).
REAL_LEAKED_TOKEN_SAMPLE = (
    '<|start|>assistant<|channel|>commentary to=functions.search_knowledge_base'
    '<|constrain|>json<|message|>{"query": "Curtailment meaning", "top_k": 10}<|call|>'
)


def _fake_message(tool_calls=None, content=None):
    """Giả lập ChatCompletionMessage — chỉ set field agent/loop.py thực sự đọc
    (content, tool_calls) + model_dump() cho nhánh append vào messages history."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (tool_calls or [])
        ] or None,
    }
    return message


def _fake_tool_call(call_id: str, name: str, arguments: str):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _fake_response(message, usage_tokens=(10, 5)):
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    if usage_tokens is None:
        resp.usage = None
    else:
        resp.usage = MagicMock(prompt_tokens=usage_tokens[0], completion_tokens=usage_tokens[1])
    return resp


def test_run_agent_loop_handles_tool_calls_with_no_content():
    """Response THẬT quan sát từ 1 proxy OpenAI-compatible: message có tool_calls
    (get_weather) nhưng content=None (kèm reasoning_content không được SDK expose
    thành attribute riêng — không liên quan tới field agent/loop.py đọc). Xác nhận
    KHÔNG crash, tool được thực thi, vòng lặp tiếp tục đúng."""
    tool_call = _fake_tool_call("call_1", "get_dataset_summary", "{}")
    first_message = _fake_message(tool_calls=[tool_call], content=None)
    second_message = _fake_message(tool_calls=None, content="Dataset có 100 dòng.")

    responses = [_fake_response(first_message), _fake_response(second_message)]

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", return_value=success({"rows": 100})) as mock_exec:
        result = run_agent_loop([{"role": "user", "content": "Dataset có bao nhiêu dòng?"}], dataset_manager=MagicMock())

    assert result["answer"] == "Dataset có 100 dòng."
    assert result["hit_max_iterations"] is False
    assert len(result["trace"]) == 1
    assert result["trace"][0]["tool_name"] == "get_dataset_summary"
    mock_exec.assert_called_once()


def test_run_agent_loop_final_answer_with_content_and_no_tool_calls():
    """Trường hợp bình thường: finish_reason="stop", có content, không có tool_calls."""
    message = _fake_message(tool_calls=None, content="Câu trả lời trực tiếp.")
    with patch("agent.loop.chat_completion", return_value=_fake_response(message)):
        result = run_agent_loop([{"role": "user", "content": "Xin chào"}], dataset_manager=MagicMock())

    assert result["answer"] == "Câu trả lời trực tiếp."
    assert result["trace"] == []
    assert result["empty_content_retries"] == 0


# =============================== _tools_for_next_call — search-until-exhausted guard ===============================

def _fake_trace_entry(tool_name: str) -> dict:
    return {"iteration": 1, "tool_name": tool_name, "arguments": {}, "result": success({})}


def test_tools_for_next_call_returns_all_schemas_when_under_threshold():
    trace = [_fake_trace_entry("search_knowledge_base") for _ in range(MAX_SEARCH_KB_CALLS - 1)]
    tools = _tools_for_next_call(trace)
    assert tools == ALL_TOOL_SCHEMAS
    names = [s["function"]["name"] for s in tools]
    assert "search_knowledge_base" in names


def test_tools_for_next_call_excludes_search_kb_at_threshold():
    """Đúng MAX_SEARCH_KB_CALLS lần (không phải nhiều hơn) đã đủ để loại tool —
    ngưỡng dùng >=, không phải >."""
    trace = [_fake_trace_entry("search_knowledge_base") for _ in range(MAX_SEARCH_KB_CALLS)]
    tools = _tools_for_next_call(trace)
    names = [s["function"]["name"] for s in tools]
    assert "search_knowledge_base" not in names
    assert "calculate" in names
    assert any(s is CALCULATE_SCHEMA for s in tools)


def test_tools_for_next_call_excludes_search_kb_beyond_threshold():
    trace = [_fake_trace_entry("search_knowledge_base") for _ in range(MAX_SEARCH_KB_CALLS + 2)]
    tools = _tools_for_next_call(trace)
    names = [s["function"]["name"] for s in tools]
    assert "search_knowledge_base" not in names


def test_tools_for_next_call_only_counts_search_kb_not_other_tools():
    """Nhiều tool KHÁC search_knowledge_base không được tính vào ngưỡng — chỉ đúng
    tool_name == 'search_knowledge_base' mới bị đếm."""
    trace = (
        [_fake_trace_entry("aggregate_metric") for _ in range(5)]
        + [_fake_trace_entry("search_knowledge_base") for _ in range(MAX_SEARCH_KB_CALLS - 1)]
    )
    tools = _tools_for_next_call(trace)
    names = [s["function"]["name"] for s in tools]
    assert "search_knowledge_base" in names


# =============================== leaked internal token detection ===============================

def test_looks_like_leaked_internal_tokens_detects_real_sample():
    assert _looks_like_leaked_internal_tokens(REAL_LEAKED_TOKEN_SAMPLE) is True


def test_looks_like_leaked_internal_tokens_false_for_normal_answer():
    assert _looks_like_leaked_internal_tokens("Tỷ lệ Normal là 77,15%.") is False


def test_looks_like_leaked_internal_tokens_detects_unseen_token_variant():
    """Regex tổng quát <|...|> phải bắt được token KHÁC 4 token đã quan sát thật,
    không chỉ hardcode đúng các case đã thấy."""
    assert _looks_like_leaked_internal_tokens("<|end_of_turn|> some other variant") is True


def test_run_agent_loop_retries_then_succeeds_after_leaked_tokens():
    """Lần đầu content là token rác (có tool_calls=None) -> phải retry qua đúng
    cơ chế empty-content-retry hiện có, KHÔNG trả rác thẳng cho user. Lần 2 content
    sạch -> trả về bình thường."""
    garbage_message = _fake_message(tool_calls=None, content=REAL_LEAKED_TOKEN_SAMPLE)
    clean_message = _fake_message(tool_calls=None, content="Curtailment là cắt giảm công suất.")

    responses = [_fake_response(garbage_message), _fake_response(clean_message)]

    with patch("agent.loop.chat_completion", side_effect=responses):
        result = run_agent_loop([{"role": "user", "content": "Curtailment nghĩa là gì?"}], dataset_manager=MagicMock())

    assert result["answer"] == "Curtailment là cắt giảm công suất."
    assert result["empty_content_retries"] == 1
    assert "<|" not in result["answer"]


def test_run_agent_loop_falls_back_cleanly_when_leaked_tokens_persist_past_retry_budget():
    """Trường hợp XẤU NHẤT: content rác ở MỌI lần gọi, kể cả sau khi
    EMPTY_CONTENT_MAX_RETRIES đã hết — _call_with_empty_content_retry() phải trả
    message rác đó ra cho run_agent_loop(), nhưng run_agent_loop() PHẢI tự kiểm tra
    LẦN NỮA và thay bằng fallback_answer sạch — KHÔNG có đường nào để rác lọt ra
    ngoài làm "answer" cuối cùng, dù đã hết lượt retry."""
    garbage_message = _fake_message(tool_calls=None, content=REAL_LEAKED_TOKEN_SAMPLE)
    # (1 lần gọi đầu + EMPTY_CONTENT_MAX_RETRIES lần retry) đều trả rác
    responses = [_fake_response(garbage_message) for _ in range(EMPTY_CONTENT_MAX_RETRIES + 1)]

    with patch("agent.loop.chat_completion", side_effect=responses):
        result = run_agent_loop([{"role": "user", "content": "Curtailment nghĩa là gì?"}], dataset_manager=MagicMock())

    assert "<|" not in result["answer"]
    assert result["answer"] == (
        "Xin lỗi, hệ thống gặp sự cố tạm thời khi tạo câu trả lời. "
        "Vui lòng thử hỏi lại câu hỏi này."
    )
    assert result["empty_content_retries"] == EMPTY_CONTENT_MAX_RETRIES
