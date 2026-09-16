"""Test cấu trúc schema hợp lệ — KHÔNG gọi LLM."""
from agent.tool_schemas import ALL_TOOL_SCHEMAS
from agent.tool_registry import TOOL_NAMES


def test_all_schemas_have_required_fields():
    for schema in ALL_TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_tool_names_are_unique():
    names = [s["function"]["name"] for s in ALL_TOOL_SCHEMAS]
    assert len(names) == len(set(names))


def test_all_schema_names_match_registry():
    schema_names = {s["function"]["name"] for s in ALL_TOOL_SCHEMAS}
    registry_names = set(TOOL_NAMES)
    assert schema_names == registry_names


def test_all_tools_defined_and_registered():
    """Không hardcode số lượng cụ thể (số cũ 21 đã lỗi thời — thực tế đã lên 22 từ khi
    thêm calculate_turbine_kpis, và sẽ tiếp tục tăng mỗi khi thêm tool mới) — so sánh
    ĐỘNG len(ALL_TOOL_SCHEMAS) == len(TOOL_NAMES) thay vào đó. Kết hợp với
    test_tool_names_are_unique (ALL_TOOL_SCHEMAS không trùng tên) và
    test_all_schema_names_match_registry (2 set khớp nhau) ở trên, phép so sánh độ dài
    này là mảnh ghép còn thiếu để đảm bảo TOOL_NAMES cũng không có tên trùng lặp — set
    equality một mình không phát hiện được trùng lặp bên trong TOOL_NAMES."""
    assert len(ALL_TOOL_SCHEMAS) == len(TOOL_NAMES)
