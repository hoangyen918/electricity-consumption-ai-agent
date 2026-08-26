"""Agent loop: LLM <-> tool execution, với MAX_ITERATIONS chống vòng lặp vô hạn."""
import json
from services.openrouter_client import chat_completion
from agent.tool_schemas import ALL_TOOL_SCHEMAS
from agent.tool_registry import execute_tool
from tools.dataset_manager import DatasetManager
from config.settings import SETTINGS


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
        }
    """
    trace = []
    usage_per_call = []
    iteration = 0
    max_iter = SETTINGS.max_agent_iterations

    while iteration < max_iter:
        iteration += 1
        response = chat_completion(messages, tools=ALL_TOOL_SCHEMAS, tool_choice="auto")

        if response.usage:
            usage_per_call.append({
                "iteration": iteration,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            })

        message = response.choices[0].message

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
            messages.append({"role": "assistant", "content": message.content})
            return {
                "answer": message.content,
                "trace": trace,
                "iterations": iteration,
                "hit_max_iterations": False,
                "usage_per_call": usage_per_call,
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
    }
