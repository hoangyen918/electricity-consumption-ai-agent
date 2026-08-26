"""
Throwaway script (M0b) — so sánh DeepSeek V3.2 vs Qwen3.6 27B qua OpenRouter
để chốt CHAT_MODEL trước khi build agent. Chạy một lần rồi bỏ, không phải
logic thường trực của app.

Usage:
    python scripts/compare_models.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from config.settings import SETTINGS

MODELS = [
    ("deepseek/deepseek-v3.2", "DEEPSEEK V3.2"),
    ("qwen/qwen3.6-27b", "QWEN3.6 27B"),
]

TOOL_DATASET_SUMMARY = {
    "type": "function",
    "function": {
        "name": "get_dataset_summary",
        "description": "Lấy thông tin tổng quan về dataset đang tải, gồm số dòng và khoảng thời gian.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

TOOL_CALCULATE = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Thực hiện phép tính đơn giản.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "percentage_of"],
                },
                "value_a": {"type": "number"},
                "value_b": {"type": "number"},
            },
            "required": ["operation", "value_a", "value_b"],
        },
    },
}

QUESTIONS = [
    {
        "label": "CÂU 1",
        "text": "Giải thích ngắn gọn bằng tiếng Việt tại sao nhiệt độ ngoài trời ảnh hưởng đến mức tiêu thụ điện.",
        "tools": None,
    },
    {
        "label": "CÂU 2",
        "text": "Mức tiêu thụ điện trung bình của Zone 1 trong tháng 7 là bao nhiêu?",
        "tools": None,
    },
    {
        "label": "CÂU 3",
        "text": "Dataset hiện tại có bao nhiêu dòng?",
        "tools": [TOOL_DATASET_SUMMARY],
    },
    {
        "label": "CÂU 4",
        "text": "Tính giúp tôi 2% của 1.200.000",
        "tools": [TOOL_CALCULATE],
    },
    {
        "label": "CÂU 5",
        "text": "Theo báo cáo thường niên EVN năm 2024, ngành nào chiếm tỷ trọng điện thương phẩm lớn nhất?",
        "tools": None,
    },
]


client = OpenAI(
    api_key=SETTINGS.openrouter_api_key,
    base_url=SETTINGS.openrouter_base_url,
)


def call_model(model_id: str, question: dict) -> dict:
    """Gửi 1 câu hỏi tới model, trả về dict mô tả kết quả (text hoặc tool_call)."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": question["text"]}],
            tools=question["tools"],
            extra_headers={
                "HTTP-Referer": SETTINGS.app_referer,
                "X-Title": SETTINGS.app_title,
            },
        )
    except Exception as exc:  # throwaway script — surface errors, don't crash the whole comparison
        return {"kind": "error", "error": str(exc)}

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = tool_call.function.arguments
        return {"kind": "tool_call", "name": tool_call.function.name, "arguments": arguments}

    return {"kind": "text", "content": (message.content or "").strip()}


def format_tool_call_line(display_name: str, result: dict) -> str:
    call_str = f"{result['name']}({json.dumps(result['arguments'], ensure_ascii=False)})"
    return f"[{display_name}] → TOOL CALL: {call_str}"


def format_no_tool_call_line(display_name: str, result: dict) -> str:
    return f'[{display_name}] ⚠️ KHÔNG gọi tool, trả lời text: "{result["content"]}"'


def main():
    tool_call_flags = {model_id: {} for model_id, _ in MODELS}

    for question in QUESTIONS:
        print("=" * 54)
        print(f"{question['label']}: {question['text']}")
        print("-" * 54)

        results = {}
        for model_id, display_name in MODELS:
            results[model_id] = call_model(model_id, question)

        is_tool_question = question["tools"] is not None

        for model_id, display_name in MODELS:
            result = results[model_id]

            if result["kind"] == "error":
                print(f"[{display_name}] ⚠️ LỖI GỌI API: {result['error']}")
                continue

            if is_tool_question:
                if result["kind"] == "tool_call":
                    print(format_tool_call_line(display_name, result))
                    tool_call_flags[model_id][question["label"]] = True
                else:
                    print(format_no_tool_call_line(display_name, result))
                    tool_call_flags[model_id][question["label"]] = False
            else:
                print(f"[{display_name}]")
                print(result.get("content", ""))
                print()

        print("=" * 54)
        print()

    # === Tổng kết ===
    ds_id, qwen_id = MODELS[0][0], MODELS[1][0]

    def yn(model_id, label):
        flag = tool_call_flags[model_id].get(label)
        if flag is None:
            return "N/A"
        return "Có" if flag else "Không"

    print("=== TỔNG KẾT ===")
    print(f"{'':18}{'DeepSeek V3.2':<16}{'Qwen3.6 27B':<16}")
    print(f"{'Câu 3 gọi tool:':18}{yn(ds_id, 'CÂU 3'):<16}{yn(qwen_id, 'CÂU 3'):<16}")
    print(f"{'Câu 4 gọi tool:':18}{yn(ds_id, 'CÂU 4'):<16}{yn(qwen_id, 'CÂU 4'):<16}")
    print(f"{'Câu 2 bịa số?:':18}{'(đọc thủ công)':<16}{'(đọc thủ công)':<16}")
    print(f"{'Câu 5 bịa nguồn?:':18}{'(đọc thủ công)':<16}{'(đọc thủ công)':<16}")
    print()
    print("Chi phí ước tính: <1 cent — an toàn để chạy nhiều lần.")


if __name__ == "__main__":
    main()
