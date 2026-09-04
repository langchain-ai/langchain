"""Convert MCP tools and tool results into LangChain-native values.

Tool results follow `langchain-mcp-adapters`, so a call made through
`langchain.mcp` reaches a model in the same shape as one loaded by that
package. Tool *metadata* is richer here: it is grouped under a single `mcp`
namespace with the tool's annotations and `_meta` under `mcp.tool` and the
serving server's identity under `mcp.server`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from fastmcp.client.group import ClientGroup
from langchain_core.messages.content import (
    FileContentBlock,
    ImageContentBlock,
    TextContentBlock,
    create_file_block,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from langchain.mcp.elicitation import _call_tool_with_interrupts, _drives_interrupts

if TYPE_CHECKING:
    from fastmcp.client import Client
    from mcp.types import Tool


class _ToolCallResult(Protocol):
    """The result fields this module reads.

    FastMCP's `CallToolResult` dataclass and the MCP SDK's own pydantic
    `CallToolResult` both satisfy this, and elicitation-driven calls return the
    latter.
    """

    content: list[ContentBlock]
    structured_content: dict[str, Any] | None
    is_error: bool


ToolMessageContentBlock = TextContentBlock | ImageContentBlock | FileContentBlock
"""LangChain content blocks an MCP tool result can convert into."""


class MCPToolArtifact(TypedDict):
    """Artifact attached to the `ToolMessage` produced by an MCP tool call.

    Wrapping the structured content in a `TypedDict` leaves room for further
    MCP result fields without changing the artifact's shape.

    Attributes:
        structured_content: The `structuredContent` of the MCP tool result.
    """

    structured_content: Any


def _summarize_tool_error(tool_content: list[ToolMessageContentBlock]) -> str:
    """Build a readable message from the content blocks of a failed tool call.

    Image and file blocks are summarized by count rather than interpolated, so a
    base64 payload never lands in an exception message.
    """
    error_parts = [block["text"] for block in tool_content if block["type"] == "text"]
    if error_parts:
        return "\n".join(error_parts)
    if tool_content:
        return (
            "The MCP tool reported an error with no text content "
            f"({len(tool_content)} non-text content block(s))."
        )
    return "The MCP tool reported an error with empty content."


class _MCPToolExecutionError(ToolException):
    """An MCP tool that ran and reported failure, as `isError=True`.

    Carries the converted content blocks so `_handle_mcp_tool_error` can hand the
    server's own error detail to the model instead of ending the run.

    Deliberately narrow: only `isError=True` results raise this. Transport and
    conversion failures are not `ToolException`s, so they bypass error handling.
    """

    def __init__(self, tool_content: list[ToolMessageContentBlock]) -> None:
        super().__init__(_summarize_tool_error(tool_content))
        self.tool_content = tool_content


def _handle_mcp_tool_error(error: ToolException) -> list[ToolMessageContentBlock]:
    """Surface an MCP execution error to the model as failed tool output.

    Wired as `handle_tool_error`, so what this returns becomes a `ToolMessage`
    with `status="error"` carrying the server's own detail. Any other
    `ToolException` is re-raised.
    """
    if isinstance(error, _MCPToolExecutionError):
        if error.tool_content:
            return error.tool_content
        # An empty `ToolMessage` is a fragile shape for some providers, so
        # substitute a placeholder rather than pass no content at all.
        return [create_text_block(text=str(error))]
    raise error


def _convert_content_block(content: ContentBlock) -> ToolMessageContentBlock:
    """Convert one MCP content block into its LangChain equivalent."""
    if isinstance(content, TextContent):
        return create_text_block(text=content.text)

    if isinstance(content, ImageContent):
        return create_image_block(base64=content.data, mime_type=content.mime_type)

    if isinstance(content, AudioContent):
        msg = (
            "Converting MCP audio content to a LangChain content block is not yet "
            f"supported. Received audio with mime type: {content.mime_type}"
        )
        raise NotImplementedError(msg)

    if isinstance(content, ResourceLink):
        mime_type = content.mime_type or None
        if mime_type and mime_type.startswith("image/"):
            return create_image_block(url=content.uri, mime_type=mime_type)
        return create_file_block(url=content.uri, mime_type=mime_type)

    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return create_text_block(text=resource.text)
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mime_type or None
            if mime_type and mime_type.startswith("image/"):
                return create_image_block(base64=resource.blob, mime_type=mime_type)
            return create_file_block(base64=resource.blob, mime_type=mime_type)
        # Unreachable while the SDK's resource union holds; see the note below.
        msg = f"Unknown embedded resource type: {type(resource).__name__}"  # type: ignore[unreachable]
        raise ValueError(msg)

    # Both unions are closed at type-check time, so mypy proves these two lines
    # unreachable — and the `unreachable` ignores become unused-ignore errors the
    # moment the SDK grows a member, which is what keeps this exhaustive.
    #
    # The runtime guards still earn their place: a union is only as closed as the
    # installed `mcp`, and a caller on a newer SDK is better served by a named
    # type than by the bare `AssertionError` an `assert_never` would raise.
    msg = (  # type: ignore[unreachable]
        f"Unknown MCP content type: {type(content).__name__}. This usually means "
        "the installed `mcp` is newer than the version this adapter supports."
    )
    raise ValueError(msg)


def _convert_call_tool_result(
    result: _ToolCallResult,
) -> tuple[list[ToolMessageContentBlock], MCPToolArtifact | None]:
    """Split an MCP tool result into model-visible content and an artifact."""
    tool_content = [_convert_content_block(block) for block in result.content]

    if result.is_error:
        raise _MCPToolExecutionError(tool_content)

    artifact: MCPToolArtifact | None = None
    if result.structured_content is not None:
        artifact = MCPToolArtifact(structured_content=result.structured_content)
    return tool_content, artifact


def _tool_metadata(tool: Tool, client: Client[Any] | None) -> dict[str, Any] | None:
    """Collect the MCP tool- and server-level metadata worth keeping.

    Everything lives under a single `mcp` namespace so a consumer can tell an
    MCP tool's provenance from any other metadata on the LangChain tool, and so
    tool-level fields (`annotations`, `_meta`) stay distinct from the identity
    of the server that served the tool.
    """
    tool_meta: dict[str, Any] = {}
    if tool.annotations is not None:
        # Snake_case (no `by_alias`) reads naturally from Python, and matches
        # what `langchain-mcp-adapters` produces.
        tool_meta["annotations"] = tool.annotations.model_dump(exclude_none=True)
    if tool.meta is not None:
        # `_meta` is the MCP wire field name; keep it so the key matches the
        # protocol and the server's payload stays in one nested place.
        tool_meta["_meta"] = tool.meta

    mcp: dict[str, Any] = {}
    if tool_meta:
        mcp["tool"] = tool_meta
    # Server identity comes off the connection, not the tool — a `Tool` carries
    # no server field. It is populated while the client is connected, which is
    # the case at conversion time.
    if client is not None and client.server_info is not None:
        mcp["server"] = client.server_info.model_dump(exclude_none=True)
    return {"mcp": mcp} if mcp else None


async def as_langchain_tool(
    tool: Tool,
    client: Client[Any] | ClientGroup,
) -> BaseTool:
    """Convert one MCP tool into a LangChain tool.

    The returned tool calls the MCP tool through `client` on every invocation.
    FastMCP clients are reentrant, so the tool can open the client itself
    whether or not a connection is already held elsewhere.

    A server that needs input mid-call is answered with a LangGraph
    `interrupt()` when `client` carries the interrupt-driving sentinel handler
    that `MCPAdapter` installs — so a human answers and the call resumes, see
    `langchain.mcp.elicitation`. A client that carries a different handler (a
    caller's own) uses that handler instead, and one with no handler simply
    never gets asked. Which path a call takes is read off the client, so the
    behavior matches whatever it was armed with.

    An MCP tool that runs and reports failure reaches the model as a
    `ToolMessage` with `status="error"`, carrying the server's own error
    content, so an agent can correct itself and retry. Transport failures and
    unconvertible content propagate instead, since a model cannot act on them.

    Args:
        tool: An MCP tool, as returned by `fastmcp.Client.list_tools`.
        client: The FastMCP client to call the tool through.

    Returns:
        A LangChain tool that invokes the MCP tool asynchronously.

    Example:
        ```python
        from fastmcp import Client

        from langchain.mcp import as_langchain_tool

        client = Client("https://example.com/mcp")
        async with client:
            mcp_tools = await client.list_tools()
        tools = [await as_langchain_tool(t, client) for t in mcp_tools]
        ```
    """
    if isinstance(client, ClientGroup):
        tool_route = await client.resolve_tool(tool.name)
        requesting_client = tool_route.client
    else:
        requesting_client = client
    drives_interrupts = _drives_interrupts(requesting_client)

    async def call_tool(
        **arguments: Any,
    ) -> tuple[list[ToolMessageContentBlock], MCPToolArtifact | None]:
        """Call the captured MCP tool and convert its result."""
        result: _ToolCallResult
        async with client:
            if drives_interrupts:
                result = await _call_tool_with_interrupts(client, tool.name, arguments)
            else:
                # Preserve MCP error results for conversion into failed tool messages.
                result = await client.call_tool(tool.name, arguments, raise_on_error=False)
        return _convert_call_tool_result(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.input_schema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=_tool_metadata(tool, requesting_client),
        handle_tool_error=_handle_mcp_tool_error,
    )


__all__ = ["MCPToolArtifact", "as_langchain_tool"]
