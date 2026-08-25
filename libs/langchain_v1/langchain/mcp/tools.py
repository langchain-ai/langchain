"""Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.
"""

from typing import Any, TypedDict, get_args

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
    InjectedToolArg,
    StructuredTool,
    ToolException,
)
from langchain_core.tools.base import get_all_basemodel_annotations
from mcp import ClientSession
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, create_model

from langchain.mcp.callbacks import CallbackContext, Callbacks, _MCPCallbacks
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


def _convert_mcp_content_to_lc_block(
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
        return create_image_block(base64=content.data, mime_type=content.mimeType)

    if isinstance(content, AudioContent):
        msg = (
            "AudioContent conversion to LangChain content blocks is not yet "
            f"supported. Received audio with mime type: {content.mimeType}"
        )
        raise NotImplementedError(msg)

    if isinstance(content, ResourceLink):
        mime_type = content.mimeType or None
        if mime_type and mime_type.startswith("image/"):
            return create_image_block(url=str(content.uri), mime_type=mime_type)
        return create_file_block(url=str(content.uri), mime_type=mime_type)

    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return create_text_block(text=resource.text)
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mimeType or None
            if mime_type and mime_type.startswith("image/"):
                return create_image_block(base64=resource.blob, mime_type=mime_type)
            return create_file_block(base64=resource.blob, mime_type=mime_type)
        msg = f"Unknown embedded resource type: {type(resource).__name__}"
        raise ValueError(msg)

    msg = f"Unknown MCP content type: {type(content).__name__}"
    raise ValueError(msg)


def _convert_call_tool_result(
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
        _convert_mcp_content_to_lc_block(content) for content in call_tool_result.content
    ]

    if call_tool_result.isError:
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
    if call_tool_result.structuredContent is not None:
        artifact = MCPToolArtifact(structured_content=call_tool_result.structuredContent)

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

        list_tools_page_result = await session.list_tools(cursor=current_cursor)

        if list_tools_page_result.tools:
            all_tools.extend(list_tools_page_result.tools)

        # Pagination spec: https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination
        # compatible with None or ""
        if not list_tools_page_result.nextCursor:
            break

        current_cursor = list_tools_page_result.nextCursor
    return all_tools


def convert_mcp_tool_to_langchain_tool(
    session: ClientSession | None,
    tool: MCPTool,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
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
        callbacks: Optional callbacks for handling notifications and events
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
        mcp_callbacks = (
            callbacks.to_mcp_format(
                context=CallbackContext(server_name=server_name, tool_name=tool.name)
            )
            if callbacks is not None
            else _MCPCallbacks()
        )

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

                async with create_session(
                    effective_connection, mcp_callbacks=mcp_callbacks
                ) as tool_session:
                    await tool_session.initialize()
                    try:
                        call_tool_result = await tool_session.call_tool(
                            tool_name,
                            tool_args,
                            progress_callback=mcp_callbacks.progress_callback,
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
                    progress_callback=mcp_callbacks.progress_callback,
                )

            return call_tool_result

        return _convert_call_tool_result(await execute_tool())

    meta = getattr(tool, "meta", None)
    base = tool.annotations.model_dump() if tool.annotations is not None else {}
    meta = {"_meta": meta} if meta is not None else {}
    metadata = {**base, **meta} or None

    # Apply server name prefix if requested
    lc_tool_name = tool.name
    if tool_name_prefix and server_name:
        lc_tool_name = f"{server_name}_{tool.name}"

    return StructuredTool(
        name=lc_tool_name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=metadata,
    )


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
    server_name: str | None = None,
    tool_name_prefix: bool = False,
) -> list[BaseTool]:
    """Load all available MCP tools and convert them to LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools).

    Args:
        session: The MCP client session. If `None`, connection must be provided.
        connection: Connection config to create a new session if session is `None`.
        callbacks: Optional `Callbacks` for handling notifications and events.
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

    mcp_callbacks = (
        callbacks.to_mcp_format(context=CallbackContext(server_name=server_name))
        if callbacks is not None
        else _MCPCallbacks()
    )

    if session is None:
        # If a session is not provided, we will create one on the fly
        if connection is None:
            msg = "Either session or connection must be provided"
            raise ValueError(msg)
        async with create_session(connection, mcp_callbacks=mcp_callbacks) as tool_session:
            await tool_session.initialize()
            tools = await _list_all_tools(tool_session)
    else:
        tools = await _list_all_tools(session)

    return [
        convert_mcp_tool_to_langchain_tool(
            session,
            tool,
            connection=connection,
            callbacks=callbacks,
            server_name=server_name,
            tool_name_prefix=tool_name_prefix,
        )
        for tool in tools
    ]


def _get_injected_args(tool: BaseTool) -> list[str]:
    """Extract field names with InjectedToolArg annotation from tool schema.

    Args:
        tool: LangChain tool to inspect.

    Returns:
        List of field names marked as injected arguments.
    """

    def _is_injected_arg_type(type_: type) -> bool:
        """Check if type annotation contains InjectedToolArg."""
        return any(
            isinstance(arg, InjectedToolArg)
            or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
            for arg in get_args(type_)[1:]
        )

    return [
        field
        for field, field_info in get_all_basemodel_annotations(tool.args_schema).items()
        if _is_injected_arg_type(field_info)
    ]


def to_fastmcp(tool: BaseTool) -> FastMCPTool:
    """Convert LangChain tool to FastMCP tool.

    Args:
        tool: LangChain tool to convert.

    Returns:
        FastMCP tool equivalent.

    Raises:
        TypeError: If args_schema is not BaseModel subclass.
        NotImplementedError: If tool has injected arguments.
    """
    if not issubclass(tool.args_schema, BaseModel):
        msg = (
            "Tool args_schema must be a subclass of pydantic.BaseModel. "
            "Tools with dict args schema are not supported."
        )
        raise TypeError(msg)

    parameters = tool.tool_call_schema.model_json_schema()
    field_definitions = {
        field: (field_info.annotation, field_info)
        for field, field_info in tool.tool_call_schema.model_fields.items()
    }
    arg_model = create_model(f"{tool.name}Arguments", **field_definitions, __base__=ArgModelBase)
    fn_metadata = FuncMetadata(arg_model=arg_model)

    # We'll use an Any type for the function return type.
    # We're providing the parameters separately
    async def fn(**arguments: dict[str, Any]) -> Any:
        return await tool.ainvoke(arguments)

    injected_args = _get_injected_args(tool)
    if len(injected_args) > 0:
        msg = "LangChain tools with injected arguments are not supported"
        raise NotImplementedError(msg)

    return FastMCPTool(
        fn=fn,
        name=tool.name,
        description=tool.description,
        parameters=parameters,
        fn_metadata=fn_metadata,
        is_async=True,
    )
