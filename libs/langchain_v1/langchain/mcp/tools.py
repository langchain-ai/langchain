"""Load tools from an MCP server (via a FastMCP client) as LangChain tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages.content import (
    ContentBlock,
    create_file_block,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import StructuredTool, ToolException
from mcp.types import (
    AudioContent,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

if TYPE_CHECKING:
    from fastmcp import Client
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.transports import ClientTransport
    from langchain_core.tools import BaseTool
    from mcp.types import ContentBlock as MCPContentBlock
    from mcp.types import Tool as MCPTool

# Result artifact carrying the server's structured (JSON) output, when provided.
_STRUCTURED_CONTENT_KEY = "structured_content"


async def load_mcp_tools(client: Client[ClientTransport]) -> list[BaseTool]:
    """Load the tools exposed by an MCP server as LangChain tools.

    Connection, transport, multi-server configuration, authentication, and protocol
    negotiation are all handled by the FastMCP client; this function only converts the
    server's tools into LangChain tools.

    Args:
        client: A FastMCP [`Client`][fastmcp.Client]. Enter it as an async context
            manager (`async with client: ...`) to reuse a single connection across tool
            calls; otherwise FastMCP connects per call as needed.

    Returns:
        The server's tools as LangChain tools.
    """
    tools = await client.list_tools()
    return [convert_mcp_tool(tool, client) for tool in tools]


def convert_mcp_tool(tool: MCPTool, client: Client[ClientTransport]) -> BaseTool:
    """Convert a single MCP tool definition into a LangChain tool.

    Args:
        tool: The MCP tool definition to convert.
        client: FastMCP client used to invoke the tool.

    Returns:
        A [`StructuredTool`][langchain_core.tools.StructuredTool] wrapping the MCP tool.
    """
    tool_name = tool.name

    async def call_tool(**arguments: Any) -> tuple[list[ContentBlock], dict[str, Any] | None]:
        result = await client.call_tool(tool_name, arguments, raise_on_error=False)
        return _convert_call_result(result)

    return StructuredTool(
        name=tool_name,
        description=tool.description or "",
        args_schema=tool.input_schema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=_tool_metadata(tool),
    )


def _tool_metadata(tool: MCPTool) -> dict[str, Any] | None:
    """Collect MCP-specific metadata (annotations, `_meta`) for the LangChain tool."""
    metadata: dict[str, Any] = {}
    if tool.annotations is not None:
        metadata["annotations"] = tool.annotations.model_dump(exclude_none=True)
    if tool.meta is not None:
        metadata["_meta"] = tool.meta
    return metadata or None


def _convert_call_result(
    result: CallToolResult,
) -> tuple[list[ContentBlock], dict[str, Any] | None]:
    """Convert an MCP tool-call result into LangChain content and an artifact.

    Args:
        result: The result returned by the FastMCP client.

    Returns:
        A `(content, artifact)` tuple. `content` is a list of content blocks; `artifact`
        carries the server's `structuredContent`, if any.

    Raises:
        ToolException: If the server reported the tool call as an error.
    """
    if result.is_error:
        text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
        raise ToolException(text or "MCP tool call failed.")

    artifact: dict[str, Any] | None = None
    if result.structured_content is not None:
        artifact = {_STRUCTURED_CONTENT_KEY: result.structured_content}

    blocks = [_convert_content_block(block) for block in result.content]
    if not blocks:
        # Surface structured-only results so the model still sees a payload.
        fallback = json.dumps(result.structured_content) if result.structured_content else ""
        blocks = [create_text_block(fallback)]

    return blocks, artifact


def _convert_content_block(block: MCPContentBlock) -> ContentBlock:
    """Map an MCP content block to the equivalent LangChain content block.

    Text maps to a text block and images to an image block. Everything else — audio,
    binary blobs, resource links, and embedded non-text resources — maps to a file block
    (text resources map to a text block). No provider currently accepts audio or video in
    a tool result, so they are represented as file attachments rather than dedicated
    audio/video blocks.
    """
    if isinstance(block, TextContent):
        return create_text_block(block.text)
    if isinstance(block, ImageContent):
        return create_image_block(base64=block.data, mime_type=block.mime_type)
    if isinstance(block, AudioContent):
        return create_file_block(base64=block.data, mime_type=block.mime_type)
    if isinstance(block, ResourceLink):
        return _resource_block(mime_type=block.mime_type, url=str(block.uri))
    resource = block.resource
    if isinstance(resource, TextResourceContents):
        return create_text_block(resource.text)
    return _resource_block(mime_type=resource.mime_type, base64=resource.blob)


def _resource_block(
    *,
    mime_type: str | None,
    url: str | None = None,
    base64: str | None = None,
) -> ContentBlock:
    """Route a linked or binary resource to an image block or a file block by MIME type."""
    if (mime_type or "").split("/", 1)[0] == "image":
        return create_image_block(url=url, base64=base64, mime_type=mime_type)
    return create_file_block(url=url, base64=base64, mime_type=mime_type)
