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


def test_converted_tool_metadata_carries_every_server_field() -> None:
    """Everything the server says about a tool travels with it, under spec names.

    `title`, `outputSchema`, `icons`, and `execution` were previously dropped, which
    silently discarded the shape of a tool's structured output.
    """
    mcp_tool = MCPTool.model_validate(
        {
            "name": "search",
            "title": "Search",
            "description": "searches",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {"hits": {"type": "integer"}}},
            "icons": [{"src": "https://example.com/icon.png", "mimeType": "image/png"}],
            "annotations": {"readOnlyHint": True},
            "_meta": {"integration": "slack", "default_interrupt": True},
        }
    )

    metadata = convert_mcp_tool_to_langchain_tool(MagicMock(), mcp_tool).metadata
    assert metadata is not None

    # Namespaced so a server cannot collide with the application's own metadata keys.
    mcp = metadata["mcp"]
    assert mcp["title"] == "Search"
    assert mcp["annotations"]["readOnlyHint"] is True
    assert mcp["outputSchema"] == {
        "type": "object",
        "properties": {"hits": {"type": "integer"}},
    }
    assert mcp["icons"][0]["src"] == "https://example.com/icon.png"
    # Non-spec extensions belong in `_meta`, which is the one place a server can put
    # them and have them survive validation.
    assert mcp["_meta"] == {"integration": "slack", "default_interrupt": True}


def test_both_titles_survive_namespacing() -> None:
    """`Tool.title` and `ToolAnnotations.title` are distinct and both are kept.

    Flattening forced one to win, since both serialize to `title`. Under the namespace
    they sit at different paths, so neither has to be dropped.
    """
    mcp_tool = MCPTool.model_validate(
        {
            "name": "search",
            "title": "Tool Title",
            "inputSchema": {},
            "annotations": {"title": "Annotation Title"},
        }
    )

    metadata = convert_mcp_tool_to_langchain_tool(MagicMock(), mcp_tool).metadata
    assert metadata is not None
    assert metadata["mcp"]["title"] == "Tool Title"
    assert metadata["mcp"]["annotations"]["title"] == "Annotation Title"
