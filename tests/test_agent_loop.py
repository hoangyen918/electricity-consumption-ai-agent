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
    _is_redundant_call,
    FINAL_SYNTHESIS_PROMPT,
    _check_budget_exceeded,
)
from agent.tool_schemas import ALL_TOOL_SCHEMAS, CALCULATE_SCHEMA
from tools.tool_results import success
from config.settings import SETTINGS

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


# =============================== _is_redundant_call — group_006 non-progress detection ===============================

def test_is_redundant_call_true_for_same_tool_and_arguments_succeeded_before():
    trace = [_fake_trace_entry("get_dataset_summary")]  # arguments={}, ok=True (xem success({}))
    assert _is_redundant_call("get_dataset_summary", {}, trace) is True


def test_is_redundant_call_false_for_different_arguments():
    trace = [{"iteration": 1, "tool_name": "aggregate_metric",
              "arguments": {"column": "humidity", "month": 6}, "result": success({})}]
    assert _is_redundant_call("aggregate_metric", {"column": "humidity", "month": 12}, trace) is False


def test_is_redundant_call_false_for_different_tool_name():
    trace = [_fake_trace_entry("get_dataset_summary")]
    assert _is_redundant_call("aggregate_metric", {}, trace) is False


def test_is_redundant_call_false_when_prior_call_failed():
    """Lan goi truoc THAT BAI (ok=False) khong tinh la 'da co ket qua that' — khong
    nen coi lan goi lai la du thua, vi lan truoc chua thanh cong."""
    trace = [{"iteration": 1, "tool_name": "get_dataset_summary", "arguments": {},
              "result": {"ok": False, "error": "lỗi tạm thời"}}]
    assert _is_redundant_call("get_dataset_summary", {}, trace) is False


def test_is_redundant_call_false_on_empty_trace():
    assert _is_redundant_call("get_dataset_summary", {}, []) is False


def test_run_agent_loop_injects_notice_message_on_redundant_call():
    """Lap lai DUNG kich ban group_006 (rut gon 2 buoc): get_dataset_summary({}) goi
    2 LAN LIEN TIEP giong het -> lan 2 phai duoc coi la du thua, va 1 message nhac
    mem phai duoc chen vao messages TRUOC khi goi LLM lan tiep theo."""
    tc1 = _fake_tool_call("call_1", "get_dataset_summary", "{}")
    tc2 = _fake_tool_call("call_2", "get_dataset_summary", "{}")
    msg1 = _fake_message(tool_calls=[tc1], content=None)
    msg2 = _fake_message(tool_calls=[tc2], content=None)
    msg3 = _fake_message(tool_calls=None, content="Dataset có 100 dòng.")

    responses = [_fake_response(msg1), _fake_response(msg2), _fake_response(msg3)]
    messages = [{"role": "user", "content": "Dataset có bao nhiêu dòng?"}]

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", return_value=success({"rows": 100})):
        result = run_agent_loop(messages, dataset_manager=MagicMock())

    assert result["answer"] == "Dataset có 100 dòng."
    notice_messages = [
        m for m in messages
        if m.get("role") == "user" and "giống hệt lần trước" in m.get("content", "")
    ]
    assert len(notice_messages) == 1
    assert "get_dataset_summary" in notice_messages[0]["content"]


# =============================== Graceful termination khi het max_agent_iterations (group_006) ===============================

def test_run_agent_loop_synthesizes_answer_from_partial_data_when_hitting_max_iterations(monkeypatch):
    """Tai hien DUNG kich ban group_006 that (rut gon max_iter=3 de test nhanh, co
    che khong doi): 2 lan get_dataset_summary({}) trung lap (lan 2 du thua) roi 1 lan
    aggregate_metric THANH CONG voi du lieu that -> het ngan sach tool-call (max_iter)
    -> PHAI thu tong hop cau tra loi tu du lieu da co (KHONG con la cau xin loi co
    dinh cu), va hit_max_iterations VAN la True (ngan sach tool that su da het)."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_agent_iterations", 3)

    tc1 = _fake_tool_call("call_1", "get_dataset_summary", "{}")
    tc2 = _fake_tool_call("call_2", "get_dataset_summary", "{}")
    tc3 = _fake_tool_call("call_3", "aggregate_metric", '{"column": "humidity", "month": 6}')
    msg1 = _fake_message(tool_calls=[tc1], content=None)
    msg2 = _fake_message(tool_calls=[tc2], content=None)
    msg3 = _fake_message(tool_calls=[tc3], content=None)
    synthesis_msg = _fake_message(
        tool_calls=None,
        content="Độ ẩm trung bình tháng 6 là 68.64 (dựa trên dữ liệu đã thu thập được).",
    )

    responses = [
        _fake_response(msg1), _fake_response(msg2), _fake_response(msg3),
        _fake_response(synthesis_msg),
    ]

    def fake_execute_tool(tool_name, arguments, dataset_manager):
        if tool_name == "aggregate_metric":
            return success({"result": 68.64})
        return success({"datasets": ["tetouan.csv"]})

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", side_effect=fake_execute_tool):
        result = run_agent_loop(
            [{"role": "user", "content": "So sánh độ ẩm trung bình giữa tháng 6 và tháng 12?"}],
            dataset_manager=MagicMock(),
        )

    assert result["hit_max_iterations"] is True
    assert result["graceful_termination"] is True
    assert "68.64" in result["answer"]
    assert "Xin lỗi, yêu cầu này cần nhiều bước xử lý" not in result["answer"]
    assert len(result["trace"]) == 3


def test_run_agent_loop_falls_back_to_old_message_when_no_successful_tool_call(monkeypatch):
    """Neu KHONG co tool call nao thanh cong (that bai tu dau), giu nguyen fallback cu
    — khong co gi de tong hop, dung goi them 1 lan LLM vo ich."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_agent_iterations", 2)

    tc1 = _fake_tool_call("call_1", "aggregate_metric", "{}")
    tc2 = _fake_tool_call("call_2", "aggregate_metric", "{}")
    msg1 = _fake_message(tool_calls=[tc1], content=None)
    msg2 = _fake_message(tool_calls=[tc2], content=None)
    responses = [_fake_response(msg1), _fake_response(msg2)]

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", return_value={"ok": False, "error": "Dataset không tồn tại."}):
        result = run_agent_loop([{"role": "user", "content": "Câu hỏi?"}], dataset_manager=MagicMock())

    assert result["hit_max_iterations"] is True
    assert result["graceful_termination"] is False
    assert result["answer"] == (
        "Xin lỗi, yêu cầu này cần nhiều bước xử lý hơn giới hạn cho phép "
        "(2 bước). Vui lòng thử chia nhỏ câu hỏi."
    )


def test_run_agent_loop_falls_back_when_synthesis_call_itself_fails(monkeypatch):
    """Neu co tool call thanh cong nhung LAN GOI TONG HOP CUOI CUNG that bai (vd loi
    API sau khi het retry) -> phai roi ve fallback cu, KHONG crash, va KHONG de rac
    FINAL_SYNTHESIS_PROMPT nam lai trong messages history."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_agent_iterations", 1)

    tc1 = _fake_tool_call("call_1", "aggregate_metric", "{}")
    msg1 = _fake_message(tool_calls=[tc1], content=None)

    def raise_on_second_call(*args, **kwargs):
        raise RuntimeError("Chat completion failed after 3 attempts")

    responses = [_fake_response(msg1)]
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return responses[0]
        raise RuntimeError("Chat completion failed after 3 attempts")

    messages = [{"role": "user", "content": "Câu hỏi?"}]
    with patch("agent.loop.chat_completion", side_effect=side_effect), \
         patch("agent.loop.execute_tool", return_value=success({"result": 42})):
        result = run_agent_loop(messages, dataset_manager=MagicMock())

    assert result["hit_max_iterations"] is True
    assert result["graceful_termination"] is False
    assert result["answer"] == (
        "Xin lỗi, yêu cầu này cần nhiều bước xử lý hơn giới hạn cho phép "
        "(1 bước). Vui lòng thử chia nhỏ câu hỏi."
    )
    # FINAL_SYNTHESIS_PROMPT phai duoc don dep, khong con sot trong lich su hoi thoai
    assert not any(m.get("content") == FINAL_SYNTHESIS_PROMPT for m in messages)


def test_run_agent_loop_falls_back_when_synthesis_call_no_tool_kwarg(monkeypatch):
    """Xac nhan lan goi tong hop cuoi cung KHONG truyen 'tools' — model bi buoc phai
    tra loi bang text, khong the goi them tool nao."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_agent_iterations", 1)

    tc1 = _fake_tool_call("call_1", "aggregate_metric", "{}")
    msg1 = _fake_message(tool_calls=[tc1], content=None)
    synthesis_msg = _fake_message(tool_calls=None, content="Kết quả: 42.")

    captured_calls = []

    def side_effect(*args, **kwargs):
        captured_calls.append(kwargs)
        if len(captured_calls) == 1:
            return _fake_response(msg1)
        return _fake_response(synthesis_msg)

    with patch("agent.loop.chat_completion", side_effect=side_effect), \
         patch("agent.loop.execute_tool", return_value=success({"result": 42})):
        result = run_agent_loop([{"role": "user", "content": "Câu hỏi?"}], dataset_manager=MagicMock())

    assert result["graceful_termination"] is True
    assert len(captured_calls) == 2
    assert "tools" not in captured_calls[1]


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


# =============================== Request Budget (Phần 1/3) — _check_budget_exceeded (pure) ===============================

def test_check_budget_exceeded_returns_none_when_under_all_thresholds():
    assert _check_budget_exceeded(1.0, 100, 0.001, 1) is None


def test_check_budget_exceeded_detects_time():
    assert _check_budget_exceeded(SETTINGS.max_request_seconds + 1, 0, 0.0, 0) == "time"


def test_check_budget_exceeded_detects_tokens():
    assert _check_budget_exceeded(0.0, SETTINGS.max_request_tokens + 1, 0.0, 0) == "tokens"


def test_check_budget_exceeded_detects_cost():
    assert _check_budget_exceeded(0.0, 0, SETTINGS.max_request_cost_usd + 0.01, 0) == "cost"


def test_check_budget_exceeded_detects_tool_calls():
    assert _check_budget_exceeded(0.0, 0, 0.0, SETTINGS.max_tool_calls + 1) == "tool_calls"


def test_check_budget_exceeded_reports_first_type_in_fixed_order_when_multiple_exceeded():
    """Ca time va tokens deu vuot cung luc -> chi anh huong loai nao duoc BAO CAO (thu
    tu co dinh time -> tokens -> cost -> tool_calls), khong anh huong viec CO dung hay
    khong (dung ngay khi bat ky 1 loai vuot)."""
    result = _check_budget_exceeded(
        SETTINGS.max_request_seconds + 1, SETTINGS.max_request_tokens + 1, 0.0, 0,
    )
    assert result == "time"


# =============================== Request Budget (Phần 1/3) — run_agent_loop tích hợp ===============================

def test_run_agent_loop_stops_before_first_iteration_when_time_budget_already_exceeded(monkeypatch):
    """Ngan sach thoi gian da vuot NGAY TU DAU (vd truoc do phien da xu ly qua lau) ->
    phai dung TRUOC KHI goi LLM lan nao ca (khong co tool call nao thuc hien), roi ve
    fallback cu (khong co du lieu thanh cong de tong hop) — nhung PHAI phan biet duoc
    voi truong hop het max_agent_iterations that su qua hit_max_iterations=False +
    budget_exceeded='time'."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_request_seconds", -1.0)

    with patch("agent.loop.chat_completion") as mock_chat:
        result = run_agent_loop([{"role": "user", "content": "Câu hỏi?"}], dataset_manager=MagicMock())

    mock_chat.assert_not_called()
    assert result["iterations"] == 0
    assert result["hit_max_iterations"] is False
    assert result["budget_exceeded"] == "time"
    assert result["graceful_termination"] is False
    assert result["trace"] == []


def test_run_agent_loop_stops_early_when_tool_calls_budget_exceeded_with_successful_data(monkeypatch):
    """max_tool_calls=1 nhung 1 lan goi LLM tra ve 2 tool_calls cung luc (xac nhan that
    o Buoc 0: 1 iteration co the chua NHIEU tool_calls) -> tong total_tool_calls=2>1
    ngay sau iteration 1 -> phai dung TRUOC iteration 2 (du max_agent_iterations mac
    dinh la 5), va van tong hop duoc cau tra loi tu du lieu da thanh cong (graceful
    termination tai su dung dung co che cu)."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_tool_calls", 1)

    tc1 = _fake_tool_call("call_1", "aggregate_metric", '{"month": 6}')
    tc2 = _fake_tool_call("call_2", "aggregate_metric", '{"month": 12}')
    msg1 = _fake_message(tool_calls=[tc1, tc2], content=None)
    synthesis_msg = _fake_message(tool_calls=None, content="Tổng hợp: tháng 6 là 68.64.")

    responses = [_fake_response(msg1), _fake_response(synthesis_msg)]

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", return_value=success({"result": 68.64})):
        result = run_agent_loop([{"role": "user", "content": "Câu hỏi?"}], dataset_manager=MagicMock())

    assert result["iterations"] == 1
    assert result["iterations"] < SETTINGS.max_agent_iterations
    assert result["hit_max_iterations"] is False
    assert result["budget_exceeded"] == "tool_calls"
    assert result["graceful_termination"] is True
    assert "68.64" in result["answer"]
    assert len(result["trace"]) == 2


def test_run_agent_loop_stops_early_when_cost_budget_exceeded_with_successful_data(monkeypatch):
    """max_request_cost_usd rat thap -> vuot ngay sau 1 lan goi LLM co usage tokens lon
    -> dung SOM, van tong hop duoc cau tra loi tu tool call da thanh cong. Dung DUNG
    calculate_chat_cost() that (khong mock) voi chat_model mac dinh (qwen/qwen3.6-27b,
    co trong MODEL_PRICING) de xac nhan cong don chi phi dung ham co san, khong viet
    lai cong thuc rieng. chat_model duoc ep ve model CO trong MODEL_PRICING vi
    SETTINGS.chat_model thuc te (tu .env, vd ag/gpt-oss-120b-medium) KHONG co trong
    bang gia -> calculate_chat_cost() se luon tra 0.0, khong the test duoc ngan sach
    cost bang model that dang chay (han che co san, ngoai pham vi task nay)."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_request_cost_usd", 0.001)
    monkeypatch.setattr("agent.loop.SETTINGS.chat_model", "qwen/qwen3.6-27b")

    tc1 = _fake_tool_call("call_1", "aggregate_metric", "{}")
    msg1 = _fake_message(tool_calls=[tc1], content=None)
    synthesis_msg = _fake_message(tool_calls=None, content="Kết quả: 42.")

    # usage lon: 1000/1e6*0.30 + 1000/1e6*2.00 = 0.0023 USD > 0.001 threshold
    responses = [
        _fake_response(msg1, usage_tokens=(1000, 1000)),
        _fake_response(synthesis_msg),
    ]

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", return_value=success({"result": 42})):
        result = run_agent_loop([{"role": "user", "content": "Câu hỏi?"}], dataset_manager=MagicMock())

    assert result["iterations"] == 1
    assert result["iterations"] < SETTINGS.max_agent_iterations
    assert result["hit_max_iterations"] is False
    assert result["budget_exceeded"] == "cost"
    assert result["graceful_termination"] is True
    assert "42" in result["answer"]


def test_run_agent_loop_stops_early_when_token_budget_exceeded_before_iteration_cap(monkeypatch):
    """Kich ban tich hop chinh yeu cau boi task: ngan sach token bi vuot SAU khi da
    chay 2 iteration THANH CONG (vd tool tra ve du lieu lon o iteration 2) — voi
    max_agent_iterations mac dinh la 5 (KHONG doi), xac nhan dung SOM HON, KHONG doi
    het ca 5 iteration."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_request_tokens", 1500)

    tc1 = _fake_tool_call("call_1", "aggregate_metric", '{"month": 6}')
    tc2 = _fake_tool_call("call_2", "aggregate_metric", '{"month": 12}')
    msg1 = _fake_message(tool_calls=[tc1], content=None)
    msg2 = _fake_message(tool_calls=[tc2], content=None)
    synthesis_msg = _fake_message(tool_calls=None, content="Tổng hợp: 42.")

    responses = [
        _fake_response(msg1, usage_tokens=(600, 600)),   # cong don 1200, <=1500, chua vuot
        _fake_response(msg2, usage_tokens=(600, 600)),   # cong don 2400, >1500, vuot
        _fake_response(synthesis_msg),
    ]

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", return_value=success({"result": 42})):
        result = run_agent_loop([{"role": "user", "content": "Câu hỏi?"}], dataset_manager=MagicMock())

    assert result["iterations"] == 2
    assert result["iterations"] < SETTINGS.max_agent_iterations
    assert result["hit_max_iterations"] is False
    assert result["budget_exceeded"] == "tokens"
    assert result["graceful_termination"] is True
    assert "42" in result["answer"]
    assert len(result["trace"]) == 2


def test_run_agent_loop_budget_exceeded_is_none_when_hitting_normal_iteration_cap(monkeypatch):
    """Doi chung: khi dung do het max_agent_iterations THAT SU (khong budget nao vuot),
    budget_exceeded phai la None va hit_max_iterations van True — khong duoc lam sai
    lech hanh vi cu (test nay lap lai kich ban group_006 da co, chi them assertion cho
    field moi budget_exceeded)."""
    monkeypatch.setattr("agent.loop.SETTINGS.max_agent_iterations", 2)

    tc1 = _fake_tool_call("call_1", "aggregate_metric", "{}")
    tc2 = _fake_tool_call("call_2", "aggregate_metric", "{}")
    msg1 = _fake_message(tool_calls=[tc1], content=None)
    msg2 = _fake_message(tool_calls=[tc2], content=None)
    responses = [_fake_response(msg1), _fake_response(msg2)]

    with patch("agent.loop.chat_completion", side_effect=responses), \
         patch("agent.loop.execute_tool", return_value={"ok": False, "error": "Dataset không tồn tại."}):
        result = run_agent_loop([{"role": "user", "content": "Câu hỏi?"}], dataset_manager=MagicMock())

    assert result["hit_max_iterations"] is True
    assert result["budget_exceeded"] is None
    assert result["graceful_termination"] is False
