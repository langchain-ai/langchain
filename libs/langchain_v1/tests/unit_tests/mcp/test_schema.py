"""Schema compatibility at the MCP tool conversion boundary."""

from copy import deepcopy
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp.types import Tool

from langchain.mcp import as_langchain_tool
from langchain.mcp.tools import _normalize_mcp_schema


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "object", "properties": {}},
        {"type": "object"},
        {"type": ["object", "null"], "properties": {}},
    ],
)
async def test_open_payload_survives_provider_conversion(payload: dict[str, Any]) -> None:
    schema = {
        "type": "object",
        "properties": {"payload": payload},
        "required": ["payload"],
    }
    original = deepcopy(schema)
    mcp_tool = Tool(name="call_operation", input_schema=schema)
    server: FastMCP[None] = FastMCP("test")
    client: Client[Any] = Client(server)
    tool = await as_langchain_tool(mcp_tool, client)
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]
    assert parameters["properties"]["payload"]["additionalProperties"] is True
    assert parameters["properties"]["payload"]["type"] == payload["type"]
    assert parameters["required"] == ["payload"]
    assert mcp_tool.input_schema == original
    assert schema == original


@pytest.mark.parametrize("additional", [False, True, {"type": "string"}])
def test_explicit_constraints_are_preserved(additional: object) -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": additional}
    assert _normalize_mcp_schema(schema) == schema


def test_nested_schemas_preserve_literal_data() -> None:
    empty = {"type": "object", "properties": {}}
    schema = {
        "$defs": {"payload": empty},
        "type": "array",
        "items": {"anyOf": [empty, {"type": "null"}]},
        "default": empty,
        "examples": [empty],
    }
    normalized = _normalize_mcp_schema(schema)
    assert normalized["$defs"]["payload"]["additionalProperties"] is True
    assert normalized["items"]["anyOf"][0]["additionalProperties"] is True
    assert normalized["default"] == empty
    assert normalized["examples"] == [empty]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {}, "unevaluatedProperties": False},
        {
            "allOf": [{"type": "object", "properties": {}}],
            "unevaluatedProperties": False,
        },
        {
            "$defs": {"payload": {"type": "object", "properties": {}}},
            "$ref": "#/$defs/payload",
            "unevaluatedProperties": {"type": "string"},
        },
    ],
)
async def test_unevaluated_properties_constraints_are_preserved(
    schema: dict[str, Any],
) -> None:
    server: FastMCP[None] = FastMCP("test")
    client: Client[Any] = Client(server)
    tool = await as_langchain_tool(Tool(name="restricted", input_schema=schema), client)
    assert tool.args_schema == schema


def test_literal_unevaluated_properties_do_not_disable_normalization() -> None:
    schema = {
        "type": "object",
        "properties": {},
        "default": {"unevaluatedProperties": False},
    }
    normalized = _normalize_mcp_schema(schema)
    assert normalized["additionalProperties"] is True
    assert normalized["default"] == schema["default"]
