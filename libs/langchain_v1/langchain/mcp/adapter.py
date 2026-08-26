"""Adapt MCP tools into LangChain tools suitable for `create_agent`."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import AnyUrl
from typing_extensions import Self

from langchain.mcp.tools import (
    _normalize_input_schema,
    _tool_error_message,
    _tool_input_schema,
    _tool_metadata,
    _tool_result_artifact,
    _tool_result_content,
)

try:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.transports import (
        ClientTransport,
        SSETransport,
        StreamableHttpTransport,
    )
    from fastmcp.exceptions import ToolError
    from fastmcp.mcp_config import MCPConfig, infer_transport_type_from_url
except ImportError as _import_error:
    msg = (
        "Please install the fastmcp client to use `MCPAdapter` — "
        '`pip install "fastmcp-slim[client]"`.'
    )
    raise ImportError(msg) from _import_error

try:
    # FastMCP 1.x servers shipped inside the MCP SDK as `mcp.server.fastmcp`. The
    # SDK renamed it to `MCPServer` in mcp 2.x (required by FastMCP 4), so this is
    # a best-effort import: it only feeds the `MCPAdapterTarget` alias, and
    # `fastmcp.Client` still infers a transport for whichever object arrives.
    from mcp.server.fastmcp import FastMCP as FastMCP1Server
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP1Server
    except ImportError:
        FastMCP1Server = Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from fastmcp import FastMCP
    from fastmcp.client.elicitation import ElicitationHandler
else:
    # In-process FastMCP servers live in the server half of FastMCP, which the
    # lightweight `fastmcp-slim[client]` install does not pull in. FastMCP degrades
    # this annotation the same way in `fastmcp.client.transports.inference`.
    FastMCP = Any


MCPAdapterTarget: TypeAlias = (
    FastMCPClient[Any]
    | ClientTransport
    | FastMCP
    | FastMCP1Server
    | AnyUrl
    | Path
    | MCPConfig
    | dict[str, Any]
    | str
)
"""Anything `MCPAdapter` accepts as its `target` argument.

Mirrors the targets `fastmcp.Client` itself accepts — a `ClientTransport`, an
in-process FastMCP server, a URL, a script `Path`, an `MCPConfig` (or its dict
form), or a string that FastMCP infers a transport from — plus a pre-built
`fastmcp.Client`.
"""


def _client_target(target: Any) -> Any:
    """Resolve URL strings to a transport before handing the target to FastMCP.

    FastMCP infers a transport by first checking whether the target names an
    existing script, which stats the filesystem. Adapters are routinely built
    inside a running event loop, so URLs take FastMCP's own URL inference
    directly rather than blocking on a filesystem call that cannot match. Every
    other target is passed through untouched for FastMCP to infer.
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

    Transport inference is delegated to `fastmcp.Client`, so a target may be a URL,
    a local script path (launched over stdio), an in-process server, or an already
    constructed client.

    Args:
        target: MCP target accepted by `fastmcp.Client`, including an existing
            FastMCP client.
        elicitation_handler: FastMCP elicitation handler invoked when a server
            requests input mid-call. Without one, FastMCP declines elicitation
            requests. Cannot be combined with a pre-built client, which carries
            its own handler.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.mcp import MCPAdapter

        async with MCPAdapter("https://example.com/mcp") as adapter:
            agent = create_agent("anthropic:claude-sonnet-5", await adapter.get_tools())
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
        ```
    """

    def __init__(
        self,
        target: MCPAdapterTarget,
        *,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> None:
        """Initialize the adapter around a FastMCP target or client."""
        if isinstance(target, FastMCPClient):
            if elicitation_handler is not None:
                msg = (
                    "`elicitation_handler` cannot be combined with a pre-built "
                    "`fastmcp.Client`. Pass the handler to the client instead."
                )
                raise ValueError(msg)
            self._client: FastMCPClient[Any] = target
        else:
            self._client = FastMCPClient(
                _client_target(target), elicitation_handler=elicitation_handler
            )
        self._lifecycle_lock = asyncio.Lock()
        self._active_uses = 0
        self._closed = False

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

    def _ensure_open(self) -> None:
        """Raise a consistent error before using an explicitly closed adapter."""
        if self._closed:
            msg = "MCPAdapter is closed and cannot be used. Create a new adapter instead."
            raise RuntimeError(msg)


__all__ = ["MCPAdapter", "MCPAdapterTarget"]
