"""Agent loop: LLM <-> tool execution, với MAX_ITERATIONS chống vòng lặp vô hạn."""
import json
import re
from services.openrouter_client import chat_completion
from agent.tool_schemas import ALL_TOOL_SCHEMAS
from agent.tool_registry import execute_tool
from tools.dataset_manager import DatasetManager
from config.settings import SETTINGS

EMPTY_CONTENT_MAX_RETRIES = 2

# Quan sát thật: gpt-oss (qua tunnel mới) đôi khi lộ token định dạng nội bộ của
# chính nó ra message.content thay vì được proxy/SDK parse đúng thành tool_calls —
# vd '<|start|>assistant<|channel|>commentary to=functions.search_knowledge_base
# <|constrain|>json<|message|>{"query": "..."}<|call|>'. Tái hiện được cả khi guard
# search-limit (MAX_SEARCH_KB_CALLS ở trên) CHƯA kích hoạt (chỉ 1 lần search đã gọi)
# — xác nhận đây là lỗi định dạng độc lập của model/proxy, KHÔNG phải do guard đó
# gây ra. Regex tổng quát <|...|> (không liệt kê cứng 4 token đã thấy) để bắt được
# cả biến thể token khác chưa từng quan sát.
_LEAKED_TOKEN_PATTERN = re.compile(r"<\|[a-z_]+\|>")


def _looks_like_leaked_internal_tokens(content: str) -> bool:
    """Phát hiện token nội bộ model bị lộ ra content (thay vì được parse đúng thành
    tool_calls) — dùng regex tổng quát <|...|> thay vì liệt kê cứng token đã quan
    sát, để bắt được cả biến thể chưa từng thấy."""
    return bool(_LEAKED_TOKEN_PATTERN.search(content))


# Guard CỨNG chống "search-until-exhausted" — quan sát thật: model gpt-oss (qua
# tunnel mới) lặp gọi search_knowledge_base 4-5 lần với query reformulate khác nhau
# rồi hit MAX_AGENT_ITERATIONS mà KHÔNG BAO GIỜ trả lời, dù đã tìm được vài kết quả
# từ những lần gọi trước đó (2/3 lần tái hiện thật). Chỉ sửa system prompt (nhắc mềm
# "không gọi quá 3 lần") KHÔNG đủ — vẫn fail 2/3 lần verify thật. Guard này chặn ở
# CẤU TRÚC: sau khi đã gọi search_knowledge_base đủ MAX_SEARCH_KB_CALLS lần, loại
# hẳn tool này khỏi danh sách "tools" gửi cho LLM ở vòng tiếp theo — LLM không còn
# lựa chọn nào khác ngoài dùng tool KHÁC (vd calculate cho câu hỏi "both") hoặc trả
# lời/từ chối bằng text. KHÔNG dùng tool_choice="none" (sẽ chặn LUÔN cả các tool khác
# như calculate, phá vỡ câu hỏi "both" cần search KB rồi tính toán tiếp).
MAX_SEARCH_KB_CALLS = 3


def _tools_for_next_call(trace: list[dict]) -> list[dict]:
    """Trả về ALL_TOOL_SCHEMAS như bình thường, TRỪ search_knowledge_base nếu đã bị
    gọi >= MAX_SEARCH_KB_CALLS lần trong trace TÍNH TỚI THỜI ĐIỂM HIỆN TẠI (không
    phải tổng cuối cùng — trace được truyền vào ngay trước mỗi lần gọi LLM, nên luôn
    phản ánh đúng số lần đã gọi CHO TỚI LÚC NÀY)."""
    search_kb_count = sum(1 for t in trace if t.get("tool_name") == "search_knowledge_base")
    if search_kb_count >= MAX_SEARCH_KB_CALLS:
        return [s for s in ALL_TOOL_SCHEMAS if s["function"]["name"] != "search_knowledge_base"]
    return ALL_TOOL_SCHEMAS


def _call_with_empty_content_retry(messages: list, tools: list[dict]) -> tuple:
    """
    Gọi chat_completion(), nếu response không có tool_calls VÀ content rỗng/None
    HOẶC chứa token nội bộ bị lộ (provider trả 'thành công' nhưng nội dung vô dụng),
    tự động thử lại tối đa EMPTY_CONTENT_MAX_RETRIES lần TRƯỚC KHI trả về cho caller
    — tách biệt hoàn toàn khỏi MAX_AGENT_ITERATIONS (ngân sách đó dành cho vòng lặp
    gọi tool, không phải cho việc phục hồi từ lỗi provider).

    tools: danh sách tool schema truyền vào lần gọi này — do caller (run_agent_loop)
    tính toán qua _tools_for_next_call(), có thể đã loại search_knowledge_base nếu
    vượt ngưỡng MAX_SEARCH_KB_CALLS.

    Trả về (response, message, empty_retries_used).
    """
    empty_retries = 0
    while True:
        response = chat_completion(messages, tools=tools, tool_choice="auto")
        message = response.choices[0].message
        has_content = (
            message.content
            and message.content.strip()
            and not _looks_like_leaked_internal_tokens(message.content)
        )
        if message.tool_calls or has_content:
            return response, message, empty_retries
        if empty_retries >= EMPTY_CONTENT_MAX_RETRIES:
            return response, message, empty_retries
        empty_retries += 1
        print(f"  [empty-content-retry] Response rỗng/lộ token nội bộ, thử lại ({empty_retries}/{EMPTY_CONTENT_MAX_RETRIES})...")


def run_agent_loop(messages: list, dataset_manager: DatasetManager) -> dict:
    """
    Chạy vòng lặp agent trên messages (list đã có system + user messages).
    MUTATES messages list trực tiếp (append tool calls + results + final answer).

    Returns:
        {
            "answer": str,
            "trace": list[dict],   # mỗi entry là 1 tool call: name, arguments, result
            "iterations": int,
            "hit_max_iterations": bool,
            "usage_per_call": list[dict],  # input/output tokens mỗi lần gọi LLM, cho M8 dùng
            "empty_content_retries": int,  # số lần phải retry do provider trả content rỗng
        }
    """
    trace = []
    usage_per_call = []
    iteration = 0
    max_iter = SETTINGS.max_agent_iterations
    total_empty_retries = 0

    while iteration < max_iter:
        iteration += 1
        tools_for_this_call = _tools_for_next_call(trace)
        response, message, empty_retries = _call_with_empty_content_retry(messages, tools_for_this_call)
        total_empty_retries += empty_retries

        if response.usage:
            usage_per_call.append({
                "iteration": iteration,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            })

        if message.tool_calls:
            messages.append(message.model_dump(exclude_unset=True))

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                result = execute_tool(tool_name, arguments, dataset_manager)

                trace.append({
                    "iteration": iteration,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            continue

        else:
            # Kiểm tra LẦN NỮA ở đây (không chỉ trong _call_with_empty_content_retry())
            # — nếu retry đã hết lượt (EMPTY_CONTENT_MAX_RETRIES) mà content VẪN rỗng
            # hoặc VẪN lộ token nội bộ, _call_with_empty_content_retry() trả message đó
            # thẳng ra cho đây, KHÔNG raise lỗi. Nếu chỉ kiểm tra content rỗng ở đây
            # (như code cũ) thì token rác (không rỗng, chỉ vô nghĩa) sẽ lọt qua nhánh
            # này và bị trả thẳng cho user ở dưới — phải kiểm tra CẢ HAI điều kiện.
            if (
                not message.content
                or not message.content.strip()
                or _looks_like_leaked_internal_tokens(message.content)
            ):
                fallback_answer = (
                    "Xin lỗi, hệ thống gặp sự cố tạm thời khi tạo câu trả lời. "
                    "Vui lòng thử hỏi lại câu hỏi này."
                )
                messages.append({"role": "assistant", "content": fallback_answer})
                return {
                    "answer": fallback_answer,
                    "trace": trace,
                    "iterations": iteration,
                    "hit_max_iterations": False,
                    "usage_per_call": usage_per_call,
                    "empty_content_retries": total_empty_retries,
                }
            messages.append({"role": "assistant", "content": message.content})
            return {
                "answer": message.content,
                "trace": trace,
                "iterations": iteration,
                "hit_max_iterations": False,
                "usage_per_call": usage_per_call,
                "empty_content_retries": total_empty_retries,
            }

    fallback_answer = (
        "Xin lỗi, yêu cầu này cần nhiều bước xử lý hơn giới hạn cho phép "
        f"({max_iter} bước). Vui lòng thử chia nhỏ câu hỏi."
    )
    messages.append({"role": "assistant", "content": fallback_answer})
    return {
        "answer": fallback_answer,
        "trace": trace,
        "iterations": iteration,
        "hit_max_iterations": True,
        "usage_per_call": usage_per_call,
        "empty_content_retries": total_empty_retries,
    }
