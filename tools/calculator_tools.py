"""Tool tính toán đơn giản — KHÔNG dùng eval(), chỉ whitelist operations."""
from tools.tool_results import success, failure

ALLOWED_OPERATIONS = ["add", "subtract", "multiply", "divide", "percentage_of", "percentage_change"]


def calculate(operation: str, value_a: float, value_b: float) -> dict:
    """
    operation: 1 trong ALLOWED_OPERATIONS
    percentage_of: value_b percent của value_a (vd: value_a=1200000, value_b=2 -> 2% của 1.2 triệu = 24000)
    percentage_change: phần trăm thay đổi từ value_a sang value_b
    """
    if operation not in ALLOWED_OPERATIONS:
        return failure(
            f"Operation '{operation}' not allowed. Valid operations: {ALLOWED_OPERATIONS}"
        )

    try:
        if operation == "add":
            result = value_a + value_b
        elif operation == "subtract":
            result = value_a - value_b
        elif operation == "multiply":
            result = value_a * value_b
        elif operation == "divide":
            if value_b == 0:
                return failure("Division by zero.")
            result = value_a / value_b
        elif operation == "percentage_of":
            result = value_a * (value_b / 100)
        elif operation == "percentage_change":
            if value_a == 0:
                return failure("Cannot calculate percentage change: base value (value_a) is zero.")
            result = ((value_b - value_a) / value_a) * 100

        return success({
            "operation": operation,
            "value_a": value_a,
            "value_b": value_b,
            "result": round(result, 6),
        })
    except Exception as e:
        return failure(f"Calculation failed: {e}")
