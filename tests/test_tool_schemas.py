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


def test_exactly_8_tools_defined():
    assert len(ALL_TOOL_SCHEMAS) == 8


def test_tool_names_are_unique():
    names = [s["function"]["name"] for s in ALL_TOOL_SCHEMAS]
    assert len(names) == len(set(names))


def test_all_schema_names_match_registry():
    schema_names = {s["function"]["name"] for s in ALL_TOOL_SCHEMAS}
    registry_names = set(TOOL_NAMES)
    assert schema_names == registry_names
