"""Tests for MCP Apps (SEP-1865) tool visibility."""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.dependencies import get_context
from mcp.types import Tool

from langchain.mcp import MCPAdapter
from langchain.mcp.apps import (
    MCP_APPS_EXTENSION,
    UI_EXTENSION_ID,
    UI_MIME_TYPE,
    app_uri,
    filter_app_visible_tools,
    filter_model_visible_tools,
    ui_meta,
)

VIEW = "ui://demo/view"


def tool(name: str, ui: dict[str, Any] | None = None) -> Tool:
    """An MCP tool definition, with the `_meta.ui` a server would declare."""
    return Tool(name=name, inputSchema={"type": "object"}, _meta={"ui": ui} if ui else None)


def test_absent_visibility_means_both_audiences() -> None:
    """The default that costs an ordinary server half its tools if inverted."""
    plain = tool("plain")

    assert filter_model_visible_tools([plain]) == [plain]
    assert filter_app_visible_tools([plain]) == [plain]


def test_a_view_without_visibility_is_still_for_both() -> None:
    """Declaring a view says nothing about who may call the tool."""
    opens_app = tool("opens_app", {"resourceUri": VIEW})

    assert filter_model_visible_tools([opens_app]) == [opens_app]
    assert filter_app_visible_tools([opens_app]) == [opens_app]


def test_an_app_only_tool_is_kept_from_the_model() -> None:
    app_only = tool("submit", {"visibility": ["app"]})

    assert filter_model_visible_tools([app_only]) == []
    assert filter_app_visible_tools([app_only]) == [app_only]


def test_a_model_only_tool_is_kept_from_the_app() -> None:
    model_only = tool("search", {"visibility": ["model"]})

    assert filter_model_visible_tools([model_only]) == [model_only]
    assert filter_app_visible_tools([model_only]) == []


def test_both_audiences_listed_is_visible_to_both() -> None:
    both = tool("either", {"visibility": ["model", "app"]})

    assert filter_model_visible_tools([both]) == [both]
    assert filter_app_visible_tools([both]) == [both]


def test_a_malformed_visibility_falls_back_to_both_not_neither() -> None:
    """A string where a list belongs must not silently hide a tool."""
    odd = tool("odd", {"visibility": "app"})

    assert filter_model_visible_tools([odd]) == [odd]
    assert filter_app_visible_tools([odd]) == [odd]


def test_filters_preserve_order_and_drop_only_what_they_must() -> None:
    tools = [tool("a"), tool("b", {"visibility": ["app"]}), tool("c")]

    assert [t.name for t in filter_model_visible_tools(tools)] == ["a", "c"]


def test_ui_meta_and_app_uri_read_a_tool_definition() -> None:
    assert ui_meta(tool("opens_app", {"resourceUri": VIEW})) == {"resourceUri": VIEW}
    assert app_uri(tool("opens_app", {"resourceUri": VIEW})) == VIEW


def test_a_tool_with_no_view_has_no_app_uri() -> None:
    assert ui_meta(tool("plain")) == {}
    assert app_uri(tool("plain")) is None
    assert app_uri(tool("submit", {"visibility": ["app"]})) is None


def test_a_non_string_resource_uri_is_not_a_uri() -> None:
    assert app_uri(tool("odd", {"resourceUri": 3})) is None


def test_the_extension_advertises_the_apps_mime_type() -> None:
    """What a server's `getUiCapability` reads to learn this host renders."""
    assert MCP_APPS_EXTENSION.identifier == UI_EXTENSION_ID == "io.modelcontextprotocol/ui"
    assert MCP_APPS_EXTENSION.settings() == {"mimeTypes": [UI_MIME_TYPE]}


async def test_a_client_sends_the_capability_to_the_server() -> None:
    """Advertising is only useful if it reaches the peer's `initialize`."""
    server = FastMCP("caps")

    @server.tool
    def client_extensions() -> str:
        """Report what the connecting client advertised."""
        params = get_context().session.client_params
        caps = params.capabilities if params else None
        return repr(getattr(caps, "extensions", None))

    async with Client(server, extensions=[MCP_APPS_EXTENSION]) as client:
        result = await client.call_tool("client_extensions")

    assert UI_EXTENSION_ID in str(result.content[0].text)


async def test_visibility_survives_the_adapter() -> None:
    """The filters have to work on adapted tools, which move `_meta`.

    An MCP `Tool` carries it on `.meta`; the same tool after `MCPAdapter`
    keeps it under `metadata["mcp"]["tool"]["_meta"]`. A host filters after
    adapting, so reading only the first shape would offer the model every
    app-only tool on the server.
    """
    server = FastMCP("apps")

    @server.tool(meta={"ui": {"resourceUri": VIEW}})
    def opens_app() -> str:
        """Ships a view, and both audiences may call it."""
        return "ok"

    @server.tool(meta={"ui": {"visibility": ["app"]}})
    def submit() -> str:
        """The app's own tool."""
        return "ok"

    async with Client(server) as client, MCPAdapter(client) as adapter:
        tools = await adapter.list_tools()

    assert {t.name for t in tools} == {"opens_app", "submit"}
    assert [t.name for t in filter_model_visible_tools(tools)] == ["opens_app"]
    assert {t.name for t in filter_app_visible_tools(tools)} == {"opens_app", "submit"}
    assert app_uri(next(t for t in tools if t.name == "opens_app")) == VIEW


if __name__ == "__main__":
    pytest.main([__file__])
