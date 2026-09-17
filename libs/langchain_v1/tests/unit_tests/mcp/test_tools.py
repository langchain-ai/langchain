"""Tests for converting MCP tools and tool results into LangChain values."""

from __future__ import annotations

import contextlib
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.client.group import ClientGroup as FastMCPClientGroup
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import ToolResult
from langgraph.errors import GraphInterrupt
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ElicitRequest,
    ElicitRequestFormParams,
    EmbeddedResource,
    ImageContent,
    InputRequiredResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)

from langchain.mcp import MCPMetaConfig, as_langchain_tool
from langchain.mcp.elicitation import _arm_for_interrupts, _call_tool_with_interrupts
from langchain.mcp.tools import (
    _convert_call_tool_result,
    _convert_content_block,
    _resolve_request_meta,
)


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
    mcp_meta: MCPMetaConfig | None = None,
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
        tool = await as_langchain_tool(mcp_tool, client, mcp_meta=mcp_meta)
    return tool, client


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
    # mcp_meta=None → response_meta disabled; FastMCP internals do not leak through
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
async def test_tool_error_surfaces_response_meta_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MCP error result preserves enabled response metadata in its artifact."""
    server: FastMCP[None] = FastMCP("flaky")

    @server.tool
    def explode() -> str:
        """Fail on purpose."""
        return "unreachable"

    tool, client = await _one_tool(server, mcp_meta=MCPMetaConfig(key="mcp_meta"))
    monkeypatch.setattr(
        client,
        "call_tool",
        AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="the widget is jammed")],
                is_error=True,
                structured_content=None,
                meta={"retryable": True},
            )
        ),
    )

    message = await tool.ainvoke(
        {"name": "explode", "args": {}, "id": "call-1", "type": "tool_call"}
    )

    assert message.status == "error"
    assert message.artifact == {
        "structured_content": None,
        "_meta": {"retryable": True},
    }


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


def test_result_with_response_meta_only_surfaces_meta() -> None:
    _, artifact = _convert_call_tool_result(
        CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content=None,
            meta={"trace_id": "abc123"},
        ),
        response_meta=True,
    )

    assert artifact is not None
    assert artifact["structured_content"] is None
    assert artifact["_meta"] == {"trace_id": "abc123"}


def test_result_with_response_meta_disabled_does_not_surface_meta() -> None:
    _, artifact = _convert_call_tool_result(
        CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content=None,
            meta={"trace_id": "abc123"},
        ),
        response_meta=False,
    )

    assert artifact is None


def test_result_with_both_structured_content_and_meta() -> None:
    _, artifact = _convert_call_tool_result(
        CallToolResult(
            content=[TextContent(type="text", text="done")],
            structured_content={"score": 0.9},
            meta={"from_cache": True},
        ),
        response_meta=True,
    )

    assert artifact == {"structured_content": {"score": 0.9}, "_meta": {"from_cache": True}}


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


# ---------------------------------------------------------------------------
# Request _meta forwarding
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_tool_with_captured_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Async factory: builds an echo tool and patches call_tool to capture kwargs.

    Returns an async callable `build(server_name)` that yields
    `(tool, captured_kwargs)`. The `captured_kwargs` list is populated by the
    patched `call_tool` on every invocation.
    """

    async def build(
        server_name: str = "tracer",
        mcp_meta: MCPMetaConfig | None = _DEFAULT_META_CFG,
    ) -> tuple[Any, Client[Any], list[dict[str, Any]]]:
        server: FastMCP[None] = FastMCP(server_name)

        @server.tool
        def echo() -> str:
            """Return ok."""
            return "ok"

        tool, client = await _one_tool(server, mcp_meta=mcp_meta)

        captured: list[dict[str, Any]] = []
        original = client.call_tool

        async def capturing(name: str, arguments: Any = None, **kwargs: Any) -> Any:
            captured.append(kwargs)
            return await original(name, arguments, **kwargs)

        monkeypatch.setattr(client, "call_tool", capturing)
        return tool, client, captured

    return build


_ECHO_CALL = {"name": "echo", "args": {}, "id": "call-echo", "type": "tool_call"}
_DEFAULT_META_CFG = MCPMetaConfig(key="mcp_meta")


@pytest.mark.asyncio
async def test_mcp_meta_configurable_is_forwarded_to_client_call_tool(
    echo_tool_with_captured_kwargs: Any,
) -> None:
    """A standalone client's configurable value is forwarded as `_meta`."""
    tool, _, captured = await echo_tool_with_captured_kwargs()

    await tool.ainvoke(
        _ECHO_CALL,
        config={
            "configurable": {
                "mcp_meta": {
                    "com.example/user-id": "u1",
                    "com.example/tenant": "acme",
                }
            }
        },
    )

    assert captured, "call_tool was never called"
    assert captured[0].get("meta") == {
        "com.example/user-id": "u1",
        "com.example/tenant": "acme",
    }


@pytest.mark.asyncio
async def test_mcp_meta_reaches_a_modern_server() -> None:
    """Application metadata reaches a modern MCP server beside protocol metadata."""
    server: FastMCP[None] = FastMCP("meta-test")

    @server.tool
    def inspect_meta(ctx: Context) -> dict[str, Any]:
        """Return the metadata observed in the active MCP request."""
        assert ctx.request_context is not None
        return {
            "protocol_version": ctx.request_context.protocol_version,
            "meta": ctx.request_context.meta,
        }

    tool, _ = await _one_tool(server, mcp_meta=MCPMetaConfig(key="mcp_meta"))

    result = await tool.ainvoke(
        {"name": "inspect_meta", "args": {}, "id": "call-1", "type": "tool_call"},
        config={
            "configurable": {
                "mcp_meta": {
                    "com.example/tenant-id": "acme",
                    "com.example/request-id": "req-123",
                }
            }
        },
    )

    assert result.artifact is not None
    payload = result.artifact["structured_content"]
    assert payload["protocol_version"] == "2026-07-28"
    assert payload["meta"]["com.example/tenant-id"] == "acme"
    assert payload["meta"]["com.example/request-id"] == "req-123"


@pytest.mark.asyncio
async def test_response_meta_from_a_modern_server_is_read_from_artifact() -> None:
    """Response `_meta` from a live modern server is exposed on the artifact."""
    server: FastMCP[None] = FastMCP("response-meta-test")

    @server.tool
    def get_result(ctx: Context) -> ToolResult:
        """Return structured data and response metadata."""
        assert ctx.request_context is not None
        return ToolResult(
            content="result",
            structured_content={
                "value": 42,
                "protocol_version": ctx.request_context.protocol_version,
            },
            meta={"com.example/source": "live-server"},
        )

    tool, _ = await _one_tool(server, mcp_meta=MCPMetaConfig(key="mcp_meta"))
    result = await tool.ainvoke(
        {"name": "get_result", "args": {}, "id": "call-1", "type": "tool_call"}
    )

    assert result.artifact is not None
    assert result.artifact["structured_content"] == {
        "value": 42,
        "protocol_version": "2026-07-28",
    }
    assert result.artifact["_meta"]["com.example/source"] == "live-server"
    assert result.artifact["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == (
        "response-meta-test"
    )


@pytest.mark.asyncio
async def test_mcp_meta_standalone_does_not_depend_on_server_identity(
    echo_tool_with_captured_kwargs: Any,
) -> None:
    """Standalone metadata remains routable after the client context closes."""
    tool, _, captured = await echo_tool_with_captured_kwargs()

    await tool.ainvoke(
        _ECHO_CALL,
        config={
            "configurable": {
                "mcp_meta": {"com.example/tenant": "direct"},
            }
        },
    )

    assert captured, "call_tool was never called"
    assert captured[0].get("meta") == {"com.example/tenant": "direct"}


@pytest.mark.asyncio
async def test_mcp_meta_per_server_uses_group_key_not_server_info_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For a ClientGroup, the caller-assigned group key (not server_info.name) is used."""
    server: FastMCP[None] = FastMCP("internal-name")

    @server.tool
    def echo() -> str:
        """Return ok."""
        return "ok"

    inner_client: Client[Any] = Client(server)
    group = FastMCPClientGroup({"my-group-key": inner_client})

    async with group:
        [mcp_tool] = await group.list_tools()
        tool = await as_langchain_tool(mcp_tool, group, mcp_meta=MCPMetaConfig(key="mcp_meta"))

    captured: list[dict[str, Any]] = []
    original = inner_client.call_tool

    async def capturing(name: str, arguments: Any = None, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return await original(name, arguments, **kwargs)

    monkeypatch.setattr(inner_client, "call_tool", capturing)

    await tool.ainvoke(
        {"name": mcp_tool.name, "args": {}, "id": "call-echo", "type": "tool_call"},
        config={
            "configurable": {
                "mcp_meta": {"my-group-key": {"com.example/tenant": "group-scoped"}},
            }
        },
    )

    assert captured, "call_tool was never called"
    assert captured[0].get("meta") == {"com.example/tenant": "group-scoped"}


@pytest.mark.asyncio
async def test_mcp_meta_standalone_forwards_an_empty_dict(
    echo_tool_with_captured_kwargs: Any,
) -> None:
    """An empty standalone metadata dict is forwarded as-is."""
    tool, _, captured = await echo_tool_with_captured_kwargs()

    await tool.ainvoke(
        _ECHO_CALL,
        config={
            "configurable": {
                "mcp_meta": {},
            }
        },
    )

    assert captured, "call_tool was never called"
    assert captured[0].get("meta") == {}


@pytest.mark.asyncio
async def test_mcp_meta_varies_per_invocation(
    echo_tool_with_captured_kwargs: Any,
) -> None:
    """mcp_meta in configurable resolves fresh on each call, enabling per-request context."""
    tool, _, captured = await echo_tool_with_captured_kwargs(mcp_meta=MCPMetaConfig(key="mcp_meta"))

    for i, tenant in enumerate(("a", "b")):
        await tool.ainvoke(
            {"name": "echo", "args": {}, "id": f"call-{i}", "type": "tool_call"},
            config={"configurable": {"mcp_meta": {"com.example/tenant": tenant}}},
        )

    assert captured[0].get("meta") == {"com.example/tenant": "a"}
    assert captured[1].get("meta") == {"com.example/tenant": "b"}


@pytest.mark.asyncio
async def test_no_mcp_meta_in_configurable_sends_none(
    echo_tool_with_captured_kwargs: Any,
) -> None:
    """When mcp_meta is absent from configurable, meta=None is passed to call_tool."""
    tool, _, captured = await echo_tool_with_captured_kwargs()

    await tool.ainvoke(_ECHO_CALL)

    assert captured, "call_tool was never called"
    assert captured[0].get("meta") is None


@pytest.mark.asyncio
async def test_invalid_mcp_meta_type_raises_type_error() -> None:
    """A non-dict value at mcp_meta_key in configurable raises TypeError."""
    server: FastMCP[None] = FastMCP("tracer")

    @server.tool
    def echo() -> str:
        """Return ok."""
        return "ok"

    tool, _ = await _one_tool(server, mcp_meta=MCPMetaConfig(key="mcp_meta"))

    with pytest.raises(TypeError, match="configurable\\['mcp_meta'\\] must be a dict or None"):
        await tool.ainvoke(
            {"name": "echo", "args": {}, "id": "c1", "type": "tool_call"},
            config={"configurable": {"mcp_meta": "not-a-dict"}},
        )


@pytest.mark.asyncio
async def test_mcp_meta_none_disables_forwarding_even_when_configurable_has_data(
    echo_tool_with_captured_kwargs: Any,
) -> None:
    """mcp_meta=None (default) ignores configurable entirely — opt-in guarantee."""
    tool, _, captured = await echo_tool_with_captured_kwargs(mcp_meta=None)

    await tool.ainvoke(
        _ECHO_CALL,
        config={"configurable": {"mcp_meta": {"tenant": "acme"}}},
    )

    assert captured, "call_tool was never called"
    assert captured[0].get("meta") is None


def test_resolve_request_meta_selects_group_member() -> None:
    config = MCPMetaConfig(key="mk")
    result = _resolve_request_meta({"mk": {"crm": {"s": 2}}}, config, "crm")
    assert result == {"s": 2}


def test_resolve_request_meta_missing_group_member_returns_none() -> None:
    config = MCPMetaConfig(key="mk")
    result = _resolve_request_meta({"mk": {"crm": {"g": 1}}}, config, "billing")
    assert result is None


def test_resolve_request_meta_request_meta_false_returns_none() -> None:
    config = MCPMetaConfig(key="mk", request_meta=False)
    result = _resolve_request_meta({"mk": {"g": 1}}, config, None)
    assert result is None


@pytest.mark.asyncio
async def test_mcp_meta_is_forwarded_through_elicitation_loop() -> None:
    """_meta is forwarded to session.call_tool inside the elicitation loop."""
    meta_value = {"tenant": "acme"}

    _elicit = InputRequiredResult(
        input_requests={
            "q1": ElicitRequest(
                method="elicitation/create",
                params=ElicitRequestFormParams(
                    message="Name?",
                    requested_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                ),
            )
        },
        request_state=None,
    )

    session_call_tool_kwargs: list[dict[str, Any]] = []

    async def fake_session_call_tool(_name: str, _arguments: Any = None, **kwargs: Any) -> Any:
        session_call_tool_kwargs.append(kwargs)
        return _elicit

    fake_session = MagicMock()
    fake_session.call_tool = fake_session_call_tool
    fake_client: Any = MagicMock()

    with (
        patch(
            "langchain.mcp.elicitation._resolve_session",
            return_value=(fake_client, fake_session, "echo"),
        ),
        patch("langchain.mcp.elicitation._await_monitored", new=lambda _c, coro: coro),
        patch("langchain.mcp.elicitation.interrupt", side_effect=GraphInterrupt(())),
        contextlib.suppress(GraphInterrupt),
    ):
        await _call_tool_with_interrupts(fake_client, "echo", {}, meta=meta_value)

    assert session_call_tool_kwargs, "session.call_tool was never called"
    assert session_call_tool_kwargs[0].get("meta") == meta_value
