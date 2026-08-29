"""Ghi log mỗi lượt hỏi-đáp vào logs/agent_trace.jsonl (JSON Lines, mỗi dòng 1 record)."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/agent_trace.jsonl")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def log_interaction(
    request_id: str,
    question: str,
    model: str,
    result: dict,
    total_latency_ms: float,
    total_cost_usd: float,
    log_path: Path = LOG_PATH,
) -> None:
    """
    result là dict trả về từ run_agent_loop(): answer, trace, iterations,
    hit_max_iterations, usage_per_call.
    log_path có thể override để test không ghi vào file thật.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "model": model,
        "iterations": result["iterations"],
        "hit_max_iterations": result["hit_max_iterations"],
        "tools_called": [
            {
                "tool_name": t["tool_name"],
                "arguments": t["arguments"],
                "ok": t["result"]["ok"],
            }
            for t in result["trace"]
        ],
        "usage_per_call": result["usage_per_call"],
        "empty_content_retries": result.get("empty_content_retries", 0),
        "total_latency_ms": round(total_latency_ms, 1),
        "estimated_cost_usd": round(total_cost_usd, 8),
        "answer_preview": (result["answer"][:200] if result["answer"] else None),
        "status": "success" if not result["hit_max_iterations"] else "max_iterations_reached",
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
