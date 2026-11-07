"""Tests for converting MCP tools and tool results into LangChain values."""

from __future__ import annotations

from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.exceptions import ToolError
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)

from langchain.mcp import as_langchain_tool
from langchain.mcp.elicitation import _arm_for_interrupts
from langchain.mcp.tools import _convert_call_tool_result, _convert_content_block


class _VideoContent:
    """Stand-in for an MCP content type newer than this adapter."""


class _VideoResource:
    """Stand-in for an embedded resource kind newer than this adapter."""


def _blocks_without_ids(content: Any) -> list[dict[str, Any]]:
    """Drop the generated block ids so content can be compared literally."""
    return [{key: value for key, value in block.items() if key != "id"} for block in content]


async def _one_tool(
    server: FastMCP[None],
    *,
    elicitation: Literal["interrupt"] | None = None,
) -> tuple[Any, Client[Any]]:
    """Convert the single tool an in-process server exposes.

    Routing is read off the client, so `elicitation='interrupt'` is exercised by
    arming the client the way the adapter would.
    """
    client: Client[Any] = Client(server)
    if elicitation == "interrupt":
        _arm_for_interrupts(client)
    async with client:
        [mcp_tool] = await client.list_tools()
    return await as_langchain_tool(mcp_tool, client), client


@pytest.mark.asyncio
async def test_text_result_becomes_content_blocks_and_structured_artifact() -> None:
    server: FastMCP[None] = FastMCP("calc")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool, _ = await _one_tool(server)

    assert tool.name == "add"
    assert tool.description == "Add two numbers."
    assert tool.args_schema["properties"] == {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
    }

    message = await tool.ainvoke(
        {"name": "add", "args": {"a": 1, "b": 2}, "id": "call-1", "type": "tool_call"}
    )

    assert _blocks_without_ids(message.content) == [{"type": "text", "text": "3"}]
    assert message.artifact == {"structured_content": {"result": 3}}
    assert message.status == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("elicitation", [None, "interrupt"], ids=["fastmcp", "interrupt"])
async def test_tool_error_reaches_the_model_as_failed_output(
    elicitation: Literal["interrupt"] | None,
) -> None:
    """An MCP `isError` result is model-visible rather than ending the run."""
    server: FastMCP[None] = FastMCP("flaky")

    @server.tool
    def explode() -> str:
        """Fail on purpose."""
        msg = "the widget is jammed"
        raise ToolError(msg)

    tool, _ = await _one_tool(server, elicitation=elicitation)

    message = await tool.ainvoke(
        {"name": "explode", "args": {}, "id": "call-1", "type": "tool_call"}
    )

    assert message.status == "error"
    [block] = _blocks_without_ids(message.content)
    assert block["type"] == "text"
    assert "the widget is jammed" in block["text"]


@pytest.mark.asyncio
async def test_client_failure_raises_instead_of_becoming_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures without an MCP result must retain normal tool error behavior."""
    server: FastMCP[None] = FastMCP("unavailable")

    @server.tool
    def ping() -> str:
        """Return a response."""
        return "pong"

    tool, client = await _one_tool(server)
    msg = "connection lost"
    monkeypatch.setattr(client, "call_tool", AsyncMock(side_effect=RuntimeError(msg)))

    with pytest.raises(RuntimeError, match=msg):
        await tool.ainvoke({"name": "ping", "args": {}, "id": "call-1", "type": "tool_call"})


def test_result_without_structured_content_has_no_artifact() -> None:
    content, artifact = _convert_call_tool_result(
        CallToolResult(
            content=[TextContent(type="text", text="hello")],
            structured_content=None,
            meta=None,
        )
    )

    assert [block["text"] for block in content if block["type"] == "text"] == ["hello"]
    assert artifact is None


@pytest.mark.asyncio
async def test_tool_stays_callable_after_its_client_context_exits() -> None:
    """FastMCP clients are reentrant, so a converted tool reconnects on demand."""
    server: FastMCP[None] = FastMCP("calc")

    @server.tool
    def double(a: int) -> int:
        """Double a number."""
        return a * 2

    tool, client = await _one_tool(server)
    assert not client.is_connected()

    message = await tool.ainvoke(
        {"name": "double", "args": {"a": 4}, "id": "c1", "type": "tool_call"}
    )

    assert _blocks_without_ids(message.content) == [{"type": "text", "text": "8"}]


@pytest.mark.asyncio
async def test_annotations_and_meta_are_kept_under_the_mcp_namespace() -> None:
    mcp_tool = Tool(
        name="delete",
        inputSchema={"type": "object", "properties": {}},
        annotations=ToolAnnotations(destructiveHint=True),
        _meta={"origin": "crm"},
    )

    tool = await as_langchain_tool(mcp_tool, Client("https://example.com/mcp"))

    # Annotations are snake_case (no wire aliases) and grouped under `mcp.tool`.
    assert tool.metadata == {
        "mcp": {"tool": {"annotations": {"destructive_hint": True}, "_meta": {"origin": "crm"}}}
    }


@pytest.mark.asyncio
async def test_tool_without_annotations_or_meta_has_no_metadata() -> None:
    mcp_tool = Tool(name="noop", inputSchema={"type": "object", "properties": {}})

    tool = await as_langchain_tool(mcp_tool, Client("https://example.com/mcp"))

    assert tool.metadata is None
    assert tool.description == ""


@pytest.mark.asyncio
async def test_server_identity_is_kept_under_the_mcp_namespace() -> None:
    server: FastMCP[None] = FastMCP("crm", version="2.1.0")

    @server.tool
    def noop() -> str:
        """Do nothing."""
        return "ok"

    # Convert while connected, as the adapter does: `server_info` is only
    # populated for the life of the client's context.
    client: Client[Any] = Client(server)
    async with client:
        [mcp_tool] = await client.list_tools()
        tool = await as_langchain_tool(mcp_tool, client)

    assert tool.metadata is not None
    assert tool.metadata["mcp"]["server"]["name"] == "crm"
    assert tool.metadata["mcp"]["server"]["version"] == "2.1.0"


def test_image_content_becomes_an_image_block() -> None:
    block = _convert_content_block(ImageContent(type="image", data="AAAA", mimeType="image/png"))

    assert block["type"] == "image"
    assert block["base64"] == "AAAA"
    assert block["mime_type"] == "image/png"


def test_text_content_becomes_a_text_block() -> None:
    block = _convert_content_block(TextContent(type="text", text="hi"))

    assert block["type"] == "text"
    assert block["text"] == "hi"


@pytest.mark.parametrize(
    ("mime_type", "expected_type"),
    [("image/png", "image"), ("application/pdf", "file"), (None, "file")],
)
def test_resource_link_type_follows_its_mime_type(
    mime_type: str | None, expected_type: str
) -> None:
    block = _convert_content_block(
        ResourceLink(
            type="resource_link",
            uri="https://example.com/report",
            name="report",
            mimeType=mime_type,
        )
    )

    assert block["type"] == expected_type
    assert block["type"] != "text"  # narrows to the blocks that carry a URL
    assert block["url"] == "https://example.com/report"


def test_embedded_text_resource_becomes_a_text_block() -> None:
    block = _convert_content_block(
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="file:///notes.txt", text="notes", mimeType="text/plain"
            ),
        )
    )

    assert block["type"] == "text"
    assert block["text"] == "notes"


@pytest.mark.parametrize(
    ("mime_type", "expected_type"),
    [("image/png", "image"), ("application/pdf", "file")],
)
def test_embedded_blob_resource_type_follows_its_mime_type(
    mime_type: str, expected_type: str
) -> None:
    block = _convert_content_block(
        EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(uri="file:///blob", blob="AAAA", mimeType=mime_type),
        )
    )

    assert block["type"] == expected_type
    assert block["type"] != "text"  # narrows to the blocks that carry base64
    assert block["base64"] == "AAAA"


def test_audio_content_is_not_yet_supported() -> None:
    with pytest.raises(NotImplementedError, match="audio"):
        _convert_content_block(AudioContent(type="audio", data="AAAA", mimeType="audio/wav"))


def test_an_unknown_content_type_names_itself_rather_than_asserting() -> None:
    """A content type the conversion has not been taught names itself.

    `ContentBlock` is closed at type-check time, but only as closed at runtime as
    the installed `mcp`. A caller on a newer SDK should learn which type arrived,
    not catch a bare `AssertionError`.
    """
    with pytest.raises(ValueError, match="Unknown MCP content type: _VideoContent"):
        _convert_content_block(cast("Any", _VideoContent()))


def test_an_unknown_embedded_resource_type_names_itself() -> None:
    """An embedded resource that is neither text nor blob raises rather than guessing."""
    embedded = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(uri="file:///notes.txt", text="notes"),
    )
    # Bypass validation to stand in for a resource kind the SDK might add later.
    object.__setattr__(embedded, "resource", cast("Any", _VideoResource()))

    with pytest.raises(ValueError, match="Unknown embedded resource type: _VideoResource"):
        _convert_content_block(embedded)
