"""Unit tests for `langchain.mcp` tool conversion.

These exercise the pure conversion logic with a stubbed client; the live transport path
is covered by the integration tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest
from langchain_core.tools import StructuredTool, ToolException
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool,
)

from langchain.mcp import convert_mcp_tool, load_mcp_tools
from langchain.mcp.tools import _convert_call_result, _convert_content_block

_ADD_CALL = {"type": "tool_call", "name": "add", "args": {"a": 2, "b": 3}, "id": "1"}

_ADD_TOOL = Tool.model_validate(
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }
)


def _result(
    *,
    content: list[Any],
    structured: dict[str, Any] | None = None,
    is_error: bool = False,
) -> Any:
    """Build a minimal stand-in for a FastMCP `CallToolResult`."""
    return SimpleNamespace(content=content, structured_content=structured, is_error=is_error)


class _StubClient:
    """Minimal stand-in for a FastMCP `Client`."""

    async def list_tools(self) -> list[Tool]:
        return [_ADD_TOOL]

    async def call_tool(self, name: str, arguments: dict[str, Any], *, raise_on_error: bool) -> Any:  # noqa: ARG002
        return _result(content=[TextContent(type="text", text="5")], structured={"result": 5})


def test_convert_content_block_text() -> None:
    block = _convert_content_block(TextContent(type="text", text="hi"))
    assert block["type"] == "text"
    assert block["text"] == "hi"


def test_convert_content_block_image_and_audio() -> None:
    image = _convert_content_block(ImageContent(type="image", data="AAA", mime_type="image/png"))
    assert image["type"] == "image"
    assert image["base64"] == "AAA"
    assert image["mime_type"] == "image/png"

    # Audio has no dedicated tool-result block across providers, so it maps to a file.
    audio = _convert_content_block(AudioContent(type="audio", data="BBB", mime_type="audio/wav"))
    assert audio["type"] == "file"
    assert audio["base64"] == "BBB"
    assert audio["mime_type"] == "audio/wav"


def test_convert_resource_link_routes_by_mime_type() -> None:
    image = _convert_content_block(
        ResourceLink(
            type="resource_link", name="i", uri="https://ex.com/i.png", mime_type="image/png"
        )
    )
    assert image["type"] == "image"
    assert "i.png" in image["url"]

    doc = _convert_content_block(
        ResourceLink(
            type="resource_link", name="d", uri="https://ex.com/d.pdf", mime_type="application/pdf"
        )
    )
    assert doc["type"] == "file"
    assert "d.pdf" in doc["url"]


def test_convert_embedded_resource() -> None:
    text = _convert_content_block(
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="file:///a.txt", text="hello", mime_type="text/plain"
            ),
        )
    )
    assert text["type"] == "text"
    assert text["text"] == "hello"

    blob = _convert_content_block(
        EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(uri="file:///a.png", blob="AAA", mime_type="image/png"),
        )
    )
    assert blob["type"] == "image"
    assert blob["base64"] == "AAA"


def test_convert_call_result_text_and_artifact() -> None:
    content, artifact = _convert_call_result(
        _result(content=[TextContent(type="text", text="5")], structured={"result": 5})
    )
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "5"
    assert artifact == {"structured_content": {"result": 5}}


def test_convert_call_result_error_raises() -> None:
    result = _result(content=[TextContent(type="text", text="boom")], is_error=True)
    with pytest.raises(ToolException, match="boom"):
        _convert_call_result(result)


async def test_load_mcp_tools_and_invoke() -> None:
    client: Any = _StubClient()
    tools = await load_mcp_tools(client)
    assert [tool.name for tool in tools] == ["add"]
    assert tools[0].description == "Add two integers."

    message = await tools[0].ainvoke(_ADD_CALL)
    assert "5" in str(message.content)


async def test_convert_mcp_tool_invocation() -> None:
    client: Any = _StubClient()
    tool = convert_mcp_tool(_ADD_TOOL, client)
    message = await tool.ainvoke(_ADD_CALL)
    assert "5" in str(message.content)


def test_tool_annotations_resolve_at_runtime() -> None:
    """`create_agent` introspects the tool coroutine with `get_type_hints`.

    Any annotation it references (e.g. `ContentBlock`) must be importable at runtime, not
    only under `TYPE_CHECKING`.
    """
    client: Any = _StubClient()
    tool = convert_mcp_tool(_ADD_TOOL, client)
    assert isinstance(tool, StructuredTool)
    assert tool.coroutine is not None
    get_type_hints(tool.coroutine)
