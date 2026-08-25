"""Unit tests for MCP tool conversion."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.types import Tool as MCPTool

from langchain.mcp.tools import (
    _normalize_input_schema,
    convert_mcp_tool_to_langchain_tool,
)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({}, {"type": "object", "properties": {}}),
        ({"type": "object"}, {"type": "object", "properties": {}}),
        ({"properties": {}}, {"type": "object", "properties": {}}),
        (
            {"type": "object", "required": ["a"]},
            {"type": "object", "properties": {}, "required": ["a"]},
        ),
    ],
)
def test_normalize_input_schema_fills_missing_defaults(
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Servers may omit `type` or `properties` for a tool that takes no arguments."""
    assert _normalize_input_schema(schema) == expected


def test_normalize_input_schema_preserves_server_values() -> None:
    """A schema that already declares everything passes through untouched."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }

    assert _normalize_input_schema(schema) == schema


def test_normalize_input_schema_does_not_override_server_type() -> None:
    """A non-object `type` is the server's bug to surface, not ours to rewrite."""
    assert _normalize_input_schema({"type": "string"}) == {
        "type": "string",
        "properties": {},
    }


def test_normalize_input_schema_does_not_mutate_input() -> None:
    """The schema belongs to the server's `Tool` object, so copy before filling in."""
    schema: dict[str, Any] = {"type": "object"}

    _normalize_input_schema(schema)

    assert schema == {"type": "object"}


@pytest.mark.parametrize("input_schema", [{}, {"type": "object"}])
def test_converted_tool_args_for_schema_without_properties(
    input_schema: dict[str, Any],
) -> None:
    """`BaseTool.args` reads `properties`, so a tool taking no arguments must carry it."""
    mcp_tool = MCPTool(name="noop", description="does nothing", inputSchema=input_schema)

    lc_tool = convert_mcp_tool_to_langchain_tool(MagicMock(), mcp_tool)

    assert lc_tool.args == {}
    assert lc_tool.args_schema == {"type": "object", "properties": {}}
