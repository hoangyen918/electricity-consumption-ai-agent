"""Ghi log mỗi lượt hỏi-đáp vào logs/agent_trace.jsonl (JSON Lines, mỗi dòng 1 record)."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/agent_trace.jsonl")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def _tool_call_entry(t: dict) -> dict:
    """
    Format 1 tool call cho tools_called. CHỈ search_knowledge_base thành công có
    thêm "retrieval_details" — list {chunk_id, source, page, distance} lấy từ
    t["result"]["data"]["results"] đã có sẵn trong trace (KHÔNG gọi lại retrieve()).
    Cố tình KHÔNG lưu "text"/nội dung chunk đầy đủ — tránh phình agent_trace.jsonl
    quá nhanh (mỗi chunk text có thể vài trăm token, x10 chunk x mỗi lượt hỏi).
    """
    entry = {
        "tool_name": t["tool_name"],
        "arguments": t["arguments"],
        "ok": t["result"]["ok"],
    }
    if t["tool_name"] == "search_knowledge_base" and t["result"]["ok"]:
        results = t["result"].get("data", {}).get("results", [])
        entry["retrieval_details"] = [
            {
                "chunk_id": r.get("chunk_id", ""),
                "source": r["source"],
                "page": r["page"],
                "distance": r["distance"],
            }
            for r in results
        ]
    return entry


def log_interaction(
    request_id: str,
    question: str,
    model: str,
    result: dict,
    total_latency_ms: float,
    total_cost_usd: float,
    log_path: Path = LOG_PATH,
    injection_flagged: bool = False,
) -> None:
    """
    result là dict trả về từ run_agent_loop(): answer, trace, iterations,
    hit_max_iterations, usage_per_call.
    log_path có thể override để test không ghi vào file thật.
    injection_flagged: kết quả agent.loop._looks_like_injection_attempt(question) —
    Lớp 2 chống prompt injection (2026-09-24), xem agent/agent.py. CHỈ để thu thập dữ
    liệu thật (tần suất gặp injection, Lớp 1+2 có đủ chặn khi bị flag hay không) —
    KHÔNG ảnh hưởng gì tới việc xử lý câu hỏi.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "model": model,
        "iterations": result["iterations"],
        "hit_max_iterations": result["hit_max_iterations"],
        "tools_called": [_tool_call_entry(t) for t in result["trace"]],
        "usage_per_call": result["usage_per_call"],
        "empty_content_retries": result.get("empty_content_retries", 0),
        "total_latency_ms": round(total_latency_ms, 1),
        "estimated_cost_usd": round(total_cost_usd, 8),
        "answer_preview": (result["answer"][:200] if result["answer"] else None),
        "status": "success" if not result["hit_max_iterations"] else "max_iterations_reached",
        "injection_flagged": injection_flagged,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
