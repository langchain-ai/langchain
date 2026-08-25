"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from typing import Any, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.messages.content import (
    FileContentBlock,
    ImageContentBlock,
    TextContentBlock,
    create_file_block,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import (
    BaseTool,
    StructuredTool,
    ToolException,
)
from mcp import ClientSession
from mcp.client.session import ProgressFnT
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PaginatedRequestParams,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool

from langchain.mcp.sessions import Connection, create_session

try:
    # langgraph installed
    from langgraph.types import Command

    LANGGRAPH_PRESENT = True
except ImportError:
    LANGGRAPH_PRESENT = False

# Type alias for LangChain content blocks used in ToolMessage
ToolMessageContentBlock = TextContentBlock | ImageContentBlock | FileContentBlock

# Conditional type based on langgraph availability
if LANGGRAPH_PRESENT:
    ConvertedToolResult = list[ToolMessageContentBlock] | ToolMessage | Command
else:
    ConvertedToolResult = list[ToolMessageContentBlock] | ToolMessage

MAX_ITERATIONS = 1000


class MCPToolArtifact(TypedDict):
    """Artifact returned from MCP tool calls.

    This TypedDict wraps the structured content from MCP tool calls,
    allowing for future extension if MCP adds more fields to tool results.

    Attributes:
        structured_content: The structured content returned by the MCP tool,
            corresponding to the structuredContent field in CallToolResult.
    """

    structured_content: dict[str, Any]


def convert_mcp_content_to_lc_block(
    content: ContentBlock,
) -> ToolMessageContentBlock:
    """Convert any MCP content block to a LangChain content block.

    Args:
        content: MCP content object (TextContent, ImageContent, AudioContent,
            ResourceLink, or EmbeddedResource).

    Returns:
        LangChain content block dict.

    Raises:
        NotImplementedError: If AudioContent is passed.
        ValueError: If an unknown content type is passed.
    """
    if isinstance(content, TextContent):
        return create_text_block(text=content.text)

    if isinstance(content, ImageContent):
        return create_image_block(base64=content.data, mime_type=content.mime_type)

    if isinstance(content, AudioContent):
        msg = (
            "AudioContent conversion to LangChain content blocks is not yet "
            f"supported. Received audio with mime type: {content.mime_type}"
        )
        raise NotImplementedError(msg)

    if isinstance(content, ResourceLink):
        mime_type = content.mime_type or None
        if mime_type and mime_type.startswith("image/"):
            return create_image_block(url=str(content.uri), mime_type=mime_type)
        return create_file_block(url=str(content.uri), mime_type=mime_type)

    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return create_text_block(text=resource.text)
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mime_type or None
            if mime_type and mime_type.startswith("image/"):
                return create_image_block(base64=resource.blob, mime_type=mime_type)
            return create_file_block(base64=resource.blob, mime_type=mime_type)
        msg = f"Unknown embedded resource type: {type(resource).__name__}"
        raise ValueError(msg)

    msg = f"Unknown MCP content type: {type(content).__name__}"
    raise ValueError(msg)


def convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[ConvertedToolResult, MCPToolArtifact | None]:
    """Convert an MCP `CallToolResult` to LangChain tool result format.

    Converts MCP content blocks to LangChain content blocks:
    - TextContent -> {"type": "text", "text": ...}
    - ImageContent -> {"type": "image", "base64": ..., "mime_type": ...}
    - ResourceLink (image/*) -> {"type": "image", "url": ..., "mime_type": ...}
    - ResourceLink (other) -> {"type": "file", "url": ..., "mime_type": ...}
    - EmbeddedResource (text) -> {"type": "text", "text": ...}
    - EmbeddedResource (blob) -> {"type": "image", ...} or {"type": "file", ...}
    - AudioContent -> raises NotImplementedError

    Args:
        call_tool_result: The result from calling an MCP tool.

    Returns:
        A tuple containing:
        - The content: either a string (single text), list of content blocks,
          ToolMessage, or Command
        - The artifact: MCPToolArtifact with structured_content if present,
          otherwise None

    Raises:
        ToolException: If the tool call resulted in an error.
        NotImplementedError: If AudioContent is encountered.
    """
    # Convert all MCP content blocks to LangChain content blocks
    tool_content: list[ToolMessageContentBlock] = [
        convert_mcp_content_to_lc_block(content) for content in call_tool_result.content
    ]

    if call_tool_result.is_error:
        # Join text from all blocks
        error_parts = []
        for item in tool_content:
            if isinstance(item, str):
                error_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                error_parts.append(item.get("text", ""))
        error_msg = "\n".join(error_parts) if error_parts else str(tool_content)
        raise ToolException(error_msg)

    # Extract structured content and wrap in MCPToolArtifact
    artifact: MCPToolArtifact | None = None
    if call_tool_result.structured_content is not None:
        artifact = MCPToolArtifact(structured_content=call_tool_result.structured_content)

    return tool_content, artifact


async def _list_all_tools(session: ClientSession) -> list[MCPTool]:
    """List all available tools from an MCP session with pagination support.

    Args:
        session: The MCP client session.

    Returns:
        A list of all available MCP tools.

    Raises:
        RuntimeError: If maximum iterations exceeded while listing tools.
    """
    current_cursor: str | None = None
    all_tools: list[MCPTool] = []

    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            msg = "Reached max of 1000 iterations while listing tools."
            raise RuntimeError(msg)

        page = PaginatedRequestParams(cursor=current_cursor) if current_cursor else None
        list_tools_page_result = await session.list_tools(params=page)

        if list_tools_page_result.tools:
            all_tools.extend(list_tools_page_result.tools)

        # Pagination spec: https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination
        # compatible with None or ""
        if not list_tools_page_result.next_cursor:
            break

        current_cursor = list_tools_page_result.next_cursor
    return all_tools


def _normalize_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Fill in structural defaults an MCP server may leave out of a tool schema.

    `type` and `properties` are both optional in JSON Schema, and servers
    legitimately omit them for a tool that takes no arguments -- `{}` and
    `{"type": "object"}` are valid ways to say "an object with no required
    fields". Provider APIs are stricter than the spec about the shape of a tool
    schema, so normalize here, at the boundary where a remote schema enters
    LangChain, rather than relying on each provider to be lenient.

    Defaults are only applied when a key is absent, so a server's own values are
    never overridden -- including a `type` other than `"object"`. A server that
    advertises a non-object schema is violating the MCP specification, and
    surfacing the provider's rejection names the offending tool, which is more
    useful than silently rewriting what the server asked for.

    Args:
        schema: The `inputSchema` advertised by the MCP server.

    Returns:
        A new schema dict. The input is never mutated, since it belongs to the
            server's `Tool` object.
    """
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


def convert_mcp_tool_to_langchain_tool(
    session: ClientSession | None,
    tool: MCPTool,
    *,
    connection: Connection | None = None,
    progress_callback: ProgressFnT | None = None,
    server_name: str | None = None,
    tool_name_prefix: bool = False,
) -> BaseTool:
    """Convert an MCP tool to a LangChain tool.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert
        connection: Optional connection config to use to create a new session
                    if a `session` is not provided
        progress_callback: Optional handler for the server's progress notifications
        server_name: Name of the server this tool belongs to
        tool_name_prefix: If `True` and `server_name` is provided, the tool name will be
            prefixed w/ server name (e.g., `"weather_search"` instead of `"search"`)

    Returns:
        a LangChain tool

    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[ConvertedToolResult, MCPToolArtifact | None]:
        """Execute the tool call and return the formatted result.

        Args:
            **arguments: Tool arguments as keyword args.

        Returns:
            A tuple of (content, artifact) where:
            - content: string, list of strings/content blocks, ToolMessage, or Command
            - artifact: MCPToolArtifact with structured_content if present, else None
        """

        async def execute_tool() -> CallToolResult:
            """Call the tool, opening a session first when one was not supplied.

            Returns:
                The `CallToolResult` from the MCP SDK.

            Raises:
                ValueError: If neither session nor connection provided.
            """
            tool_name = tool.name
            tool_args = arguments
            effective_connection = connection
            captured_exception = None

            if session is None:
                # If a session is not provided, we will create one on the fly
                if effective_connection is None:
                    msg = "Either session or connection must be provided"
                    raise ValueError(msg)

                async with create_session(effective_connection) as tool_session:
                    await tool_session.initialize()
                    try:
                        call_tool_result = await tool_session.call_tool(
                            tool_name,
                            tool_args,
                            progress_callback=progress_callback,
                        )
                    except Exception as e:
                        # Capture exception to re-raise outside context manager
                        captured_exception = e

                # Re-raise the exception outside the context manager
                # This is necessary because the context manager may suppress exceptions
                # This change was introduced to work-around an issue in MCP SDK
                # that may suppress exceptions when the client disconnects.
                # If this is causing an issue, with your use case, please file an issue
                # on the langchain-mcp-adapters GitHub repo.
                if captured_exception is not None:
                    raise captured_exception
            else:
                call_tool_result = await session.call_tool(
                    tool_name,
                    tool_args,
                    progress_callback=progress_callback,
                )

            return call_tool_result

        return convert_call_tool_result(await execute_tool())

    meta = getattr(tool, "meta", None)
    # Dump by alias so these stay the specification's names (`readOnlyHint`), which
    # mcp 2.x renamed to snake_case on the model itself.
    base = tool.annotations.model_dump(by_alias=True) if tool.annotations is not None else {}
    meta = {"_meta": meta} if meta is not None else {}
    metadata = {**base, **meta} or None

    # Apply server name prefix if requested
    lc_tool_name = tool.name
    if tool_name_prefix and server_name:
        lc_tool_name = f"{server_name}_{tool.name}"

    return StructuredTool(
        name=lc_tool_name,
        description=tool.description or "",
        args_schema=_normalize_input_schema(tool.input_schema),
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=metadata,
    )


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    progress_callback: ProgressFnT | None = None,
    server_name: str | None = None,
    tool_name_prefix: bool = False,
) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools).

    Args:
        session: The MCP client session. If `None`, connection must be provided.
        connection: Connection config to create a new session if session is `None`.
        progress_callback: Optional handler for the server's progress notifications.
        server_name: Name of the server these tools belong to.
        tool_name_prefix: If `True` and `server_name` is provided, tool names will be
            prefixed w/ server name (e.g., `"weather_search"` instead of `"search"`).

    Returns:
        List of LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools).
            Tool annotations are returned as part of the tool metadata object.

    Raises:
        ValueError: If neither session nor connection is provided.
    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    if session is None:
        # If a session is not provided, we will create one on the fly
        if connection is None:
            msg = "Either session or connection must be provided"
            raise ValueError(msg)
        async with create_session(connection) as tool_session:
            await tool_session.initialize()
            tools = await _list_all_tools(tool_session)
    else:
        tools = await _list_all_tools(session)

    return [
        convert_mcp_tool_to_langchain_tool(
            session,
            tool,
            connection=connection,
            progress_callback=progress_callback,
            server_name=server_name,
            tool_name_prefix=tool_name_prefix,
        )
        for tool in tools
    ]
