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


def test_log_interaction_includes_retrieval_details_for_search_kb(tmp_path):
    """retrieval_details phải xuất hiện CHỈ cho search_knowledge_base thành công, chứa
    đúng chunk_id/source/page/distance, KHÔNG chứa "text" (tránh phình log)."""
    log_path = tmp_path / "test_trace.jsonl"
    fake_result = {
        "answer": "Câu trả lời có trích dẫn.",
        "trace": [
            {
                "tool_name": "search_knowledge_base",
                "arguments": {"query": "tỷ lệ tổn thất điện năng"},
                "result": {
                    "ok": True,
                    "data": {
                        "results": [
                            {
                                "chunk_id": "evn_2022_2023_p010_c00",
                                "text": "Đoạn văn bản rất dài của chunk, không nên lưu vào log..." * 20,
                                "source": "EVN Annual Report 2022-2023",
                                "page": 10,
                                "citation": "[EVN Annual Report 2022-2023, p.10]",
                                "distance": 0.8123,
                            }
                        ],
                        "result_count": 1,
                    },
                },
            }
        ],
        "iterations": 1,
        "hit_max_iterations": False,
        "usage_per_call": [],
    }

    log_interaction("req_kb1", "Q?", "model", fake_result, 100, 0.0, log_path=log_path)

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    tool_entry = entry["tools_called"][0]
    assert "retrieval_details" in tool_entry
    assert tool_entry["retrieval_details"] == [
        {
            "chunk_id": "evn_2022_2023_p010_c00",
            "source": "EVN Annual Report 2022-2023",
            "page": 10,
            "distance": 0.8123,
        }
    ]
    # KHONG duoc lo full text chunk vao log
    assert "text" not in tool_entry["retrieval_details"][0]
    # Cac field cu (tool_name/arguments/ok) van nguyen, khong doi format
    assert tool_entry["tool_name"] == "search_knowledge_base"
    assert tool_entry["arguments"] == {"query": "tỷ lệ tổn thất điện năng"}
    assert tool_entry["ok"] is True


def test_log_interaction_no_retrieval_details_for_non_search_tool(tmp_path):
    """Tool khác search_knowledge_base KHONG duoc them retrieval_details — giu nguyen
    format 3-field cu (tool_name/arguments/ok)."""
    log_path = tmp_path / "test_trace.jsonl"
    fake_result = {
        "answer": "42",
        "trace": [
            {"tool_name": "aggregate_metric", "arguments": {"column": "x"}, "result": {"ok": True}}
        ],
        "iterations": 1,
        "hit_max_iterations": False,
        "usage_per_call": [],
    }
    log_interaction("req_agg1", "Q?", "model", fake_result, 100, 0.0, log_path=log_path)

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    tool_entry = entry["tools_called"][0]
    assert tool_entry == {"tool_name": "aggregate_metric", "arguments": {"column": "x"}, "ok": True}
    assert "retrieval_details" not in tool_entry


def test_log_interaction_no_retrieval_details_when_search_kb_failed(tmp_path):
    """search_knowledge_base that bai (ok=False) KHONG duoc them retrieval_details (khong
    co result["data"] de doc)."""
    log_path = tmp_path / "test_trace.jsonl"
    fake_result = {
        "answer": "Xin lỗi, không tìm thấy.",
        "trace": [
            {
                "tool_name": "search_knowledge_base",
                "arguments": {"query": "abc"},
                "result": {"ok": False, "error": "No relevant information found."},
            }
        ],
        "iterations": 1,
        "hit_max_iterations": False,
        "usage_per_call": [],
    }
    log_interaction("req_kbfail1", "Q?", "model", fake_result, 100, 0.0, log_path=log_path)

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    tool_entry = entry["tools_called"][0]
    assert "retrieval_details" not in tool_entry
    assert tool_entry["ok"] is False
