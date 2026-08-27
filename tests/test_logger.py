import json
from services.logger import new_request_id, log_interaction


def test_new_request_id_format():
    rid = new_request_id()
    assert rid.startswith("req_")
    assert len(rid) == 16  # "req_" + 12 hex chars


def test_log_interaction_writes_valid_jsonl(tmp_path):
    log_path = tmp_path / "test_trace.jsonl"

    fake_result = {
        "answer": "Câu trả lời test.",
        "trace": [
            {"tool_name": "aggregate_metric", "arguments": {"column": "x"}, "result": {"ok": True}}
        ],
        "iterations": 2,
        "hit_max_iterations": False,
        "usage_per_call": [{"iteration": 1, "input_tokens": 100, "output_tokens": 50}],
    }

    log_interaction(
        request_id="req_test123",
        question="Câu hỏi test?",
        model="qwen/qwen3.6-27b",
        result=fake_result,
        total_latency_ms=1234.5,
        total_cost_usd=0.0001,
        log_path=log_path,
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["request_id"] == "req_test123"
    assert entry["question"] == "Câu hỏi test?"
    assert entry["iterations"] == 2
    assert entry["status"] == "success"
    assert len(entry["tools_called"]) == 1


def test_log_interaction_appends_not_overwrites(tmp_path):
    log_path = tmp_path / "test_trace.jsonl"
    fake_result = {
        "answer": "A", "trace": [], "iterations": 1,
        "hit_max_iterations": False, "usage_per_call": [],
    }
    log_interaction("req_1", "Q1", "model", fake_result, 100, 0.0, log_path=log_path)
    log_interaction("req_2", "Q2", "model", fake_result, 100, 0.0, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
