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


@pytest.mark.parametrize(
    "constraint",
    [
        {"additionalProperties": False},
        {"additionalProperties": {"type": "string"}},
        {"unevaluatedProperties": False},
    ],
)
def test_explicit_constraints_are_preserved(constraint: dict[str, Any]) -> None:
    payload = {"type": "object", **constraint}
    schema = {"type": "object", "properties": {"payload": payload}}
    assert _normalize_mcp_schema(schema)["properties"]["payload"] == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "object"},
        {"type": ["object", "null"]},
    ],
)
async def test_nested_open_object_closes_under_strict(payload: dict[str, Any]) -> None:
    """A nested bare object must be closed the same way a direct one is.

    Regression test for https://github.com/langchain-ai/langchain/issues/41321.
    """
    schema = {
        "type": "object",
        "properties": {
            "wrapper": {
                "type": "object",
                "properties": {"payload": payload},
                "required": ["payload"],
            }
        },
        "required": ["wrapper"],
    }
    mcp_tool = Tool(name="call_operation", input_schema=schema)
    tool = await as_langchain_tool(mcp_tool, Client(FastMCP("test")))
    parameters = convert_to_openai_tool(tool, strict=True)["function"]["parameters"]

    nested = parameters["properties"]["wrapper"]["properties"]["payload"]
    assert nested["additionalProperties"] is False
    assert nested["type"] == payload["type"]
