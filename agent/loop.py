"""Agent loop: LLM <-> tool execution, với MAX_ITERATIONS chống vòng lặp vô hạn."""
import json
from services.openrouter_client import chat_completion
from agent.tool_schemas import ALL_TOOL_SCHEMAS
from agent.tool_registry import execute_tool
from tools.dataset_manager import DatasetManager
from config.settings import SETTINGS

EMPTY_CONTENT_MAX_RETRIES = 2


def _call_with_empty_content_retry(messages: list) -> tuple:
    """
    Gọi chat_completion(), nếu response không có tool_calls VÀ content rỗng/None
    (provider trả 'thành công' nhưng nội dung vô dụng), tự động thử lại tối đa
    EMPTY_CONTENT_MAX_RETRIES lần TRƯỚC KHI trả về cho caller — tách biệt hoàn toàn
    khỏi MAX_AGENT_ITERATIONS (ngân sách đó dành cho vòng lặp gọi tool, không phải
    cho việc phục hồi từ lỗi provider).
    Trả về (response, message, empty_retries_used).
    """
    empty_retries = 0
    while True:
        response = chat_completion(messages, tools=ALL_TOOL_SCHEMAS, tool_choice="auto")
        message = response.choices[0].message
        has_content = message.content and message.content.strip()
        if message.tool_calls or has_content:
            return response, message, empty_retries
        if empty_retries >= EMPTY_CONTENT_MAX_RETRIES:
            return response, message, empty_retries
        empty_retries += 1
        print(f"  [empty-content-retry] Response rỗng, thử lại ({empty_retries}/{EMPTY_CONTENT_MAX_RETRIES})...")


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
        response, message, empty_retries = _call_with_empty_content_retry(messages)
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
            if not message.content or not message.content.strip():
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
