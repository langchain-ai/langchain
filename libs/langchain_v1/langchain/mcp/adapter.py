"""Adapt MCP tools into LangChain tools suitable for `create_agent`."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from fastmcp import FastMCP

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from typing_extensions import Self

from langchain.mcp.tools import (
    _normalize_input_schema,
    _tool_error_message,
    _tool_input_schema,
    _tool_metadata,
    _tool_result_artifact,
    _tool_result_content,
    interrupt_for_elicitation,
)

try:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.transports import (
        ClientTransport,
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )
    from fastmcp.exceptions import ToolError
    from fastmcp.mcp_config import infer_transport_type_from_url
except ImportError as _import_error:
    msg = (
        "Please install the fastmcp client to use `MCPAdapter` — "
        '`pip install "fastmcp-slim[client]"`.'
    )
    raise ImportError(msg) from _import_error


# In-process FastMCP servers live in the server half of FastMCP. The lightweight
# `fastmcp-slim[client]` install does not ship it, so guard the import separately.
# The alias widens to `Any` in that environment; passing an in-process server is
# then impossible at runtime anyway.
if not TYPE_CHECKING:
    try:
        from fastmcp import FastMCP
    except ImportError:  # pragma: no cover
        FastMCP = Any


MCPAdapterTarget: TypeAlias = FastMCPClient[Any] | ClientTransport | FastMCP | Path | str
"""Anything `MCPAdapter` accepts as its `target` argument.

This may be a pre-built `fastmcp.Client`, a `ClientTransport`, an in-process
FastMCP server, a URL/path string, or a script `Path`.
"""


def _client_target(target: Any) -> Any:
    """Turn URL strings into transports before FastMCP performs path inference.

    FastMCP also supports script paths expressed as strings. Those are deliberately
    left alone here so its stdio inference remains available after the adapter's
    explicit stdio policy check.
    """
    if not isinstance(target, str) or not target.startswith(("http://", "https://")):
        return target
    if infer_transport_type_from_url(target) == "sse":
        return SSETransport(url=target)
    return StreamableHttpTransport(url=target)


class MCPAdapter:
    """Adapt an MCP target into LangChain tools.

    `MCPAdapter` uses FastMCP for protocol negotiation and connection management,
    then converts discovered MCP tools into asynchronous LangChain tools. The
    resulting tools can be passed directly to `create_agent`.

    Args:
        target: MCP target accepted by `fastmcp.Client`, including an existing
            FastMCP client.
        allow_stdio: Whether a target that launches a local stdio subprocess is
            allowed. Disabled by default because it executes a local program.
    """

    def __init__(
        self,
        target: MCPAdapterTarget,
        *,
        allow_stdio: bool = False,
    ) -> None:
        """Initialize the adapter around a FastMCP target or client."""
        self._validate_target(target, allow_stdio=allow_stdio)
        self._client = (
            target
            if isinstance(target, FastMCPClient)
            else FastMCPClient(_client_target(target), elicitation_handler=self._handle_elicitation)
        )
        self._lifecycle_lock = asyncio.Lock()
        self._active_uses = 0
        self._closed = False

    async def _handle_elicitation(
        self,
        _message: str,
        _response_type: type[Any] | None,
        params: Any,
        _context: Any,
    ) -> dict[str, Any]:
        """Surface FastMCP elicitation through LangGraph's interrupt mechanism."""
        return interrupt_for_elicitation(params)

    @property
    def client(self) -> FastMCPClient[Any]:
        """Return the underlying FastMCP client for advanced MCP operations."""
        return self._client

    async def __aenter__(self) -> Self:
        """Acquire a connection use for an explicit adapter context."""
        await self._acquire_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release an explicit adapter context."""
        await self._release_client(exc_type, exc_value, traceback)

    async def aclose(self) -> None:
        """Prevent new work and close when active operations have completed."""
        async with self._lifecycle_lock:
            self._closed = True

    async def get_tools(self) -> list[BaseTool]:
        """Discover and adapt MCP tools for use with LangChain.

        Returns:
            LangChain tools that invoke the corresponding MCP tools asynchronously.

        Raises:
            ValueError: If the MCP server exposes duplicate tool names.
            RuntimeError: If the adapter has been closed.
        """
        async with self._client_context():
            remote_tools = await self._client.list_tools()

        tools = [self._convert_tool(tool) for tool in remote_tools]
        self._validate_unique_names(tools)
        return tools

    @asynccontextmanager
    async def _client_context(self) -> AsyncIterator[None]:
        """Keep the client connected for one operation without racing closure."""
        await self._acquire_client()
        try:
            yield
        except BaseException as error:
            await self._release_client(type(error), error, error.__traceback__)
            raise
        else:
            await self._release_client(None, None, None)

    async def _acquire_client(self) -> None:
        """Acquire one active use, connecting the FastMCP client if needed."""
        async with self._lifecycle_lock:
            self._ensure_open()
            if self._active_uses == 0:
                await self._client.__aenter__()  # type: ignore[no-untyped-call]
            self._active_uses += 1

    async def _release_client(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release one active use and close the client after the final release."""
        async with self._lifecycle_lock:
            if self._active_uses == 0:
                return
            self._active_uses -= 1
            if self._active_uses == 0:
                await self._client.__aexit__(exc_type, exc_value, traceback)  # type: ignore[no-untyped-call]

    def _convert_tool(self, tool: Any) -> BaseTool:
        """Convert one FastMCP tool model into a LangChain `StructuredTool`."""
        remote_name = getattr(tool, "name", None)
        if not isinstance(remote_name, str) or not remote_name:
            msg = "MCP returned a tool without a non-empty string name."
            raise ValueError(msg)
        description = getattr(tool, "description", "") or ""

        async def call_tool(**arguments: Any) -> tuple[Any, dict[str, Any] | None]:
            """Invoke the captured remote tool through the managed MCP client."""
            self._ensure_open()
            try:
                async with self._client_context():
                    result = await self._client.call_tool(remote_name, arguments)
            except ToolError as error:
                raise ToolException(str(error)) from error
            if error_message := _tool_error_message(result):
                raise ToolException(error_message)
            return _tool_result_content(result), _tool_result_artifact(result)

        return StructuredTool(
            name=remote_name,
            description=description,
            args_schema=_normalize_input_schema(_tool_input_schema(tool)),
            coroutine=call_tool,
            response_format="content_and_artifact",
            metadata=_tool_metadata(tool),
        )

    @staticmethod
    def _validate_unique_names(tools: list[BaseTool]) -> None:
        """Reject ambiguous tool registries before an agent indexes them by name."""
        names = [tool.name for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"MCP returned duplicate tool names: {', '.join(duplicates)}."
            raise ValueError(msg)

    @staticmethod
    def _validate_target(target: MCPAdapterTarget, *, allow_stdio: bool) -> None:
        """Reject subprocess-spawning targets unless callers explicitly opt in."""
        if allow_stdio:
            return
        target_transport = getattr(target, "transport", target)
        if isinstance(target_transport, (Path, StdioTransport)):
            msg = "Stdio MCP targets require `allow_stdio=True`."
            raise ValueError(msg)  # noqa: TRY004
        if isinstance(target, str):
            try:
                infer_transport_type_from_url(target)
            except ValueError as error:
                msg = "Stdio MCP targets require `allow_stdio=True`."
                raise ValueError(msg) from error

    def _ensure_open(self) -> None:
        """Raise a consistent error before using an explicitly closed adapter."""
        if self._closed:
            msg = "MCPAdapter is closed and cannot be used. Create a new adapter instead."
            raise RuntimeError(msg)


__all__ = ["MCPAdapter", "MCPAdapterTarget"]
