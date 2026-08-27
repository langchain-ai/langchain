"""Adapt MCP tools into LangChain tools suitable for `create_agent`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import AnyUrl
from typing_extensions import Self

from langchain.mcp.tools import convert_mcp_tool_to_langchain_tool

try:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.transports import ClientTransport
    from fastmcp.mcp_config import MCPConfig

    # The MCP SDK's own server type, which `fastmcp.Client` also accepts
    # in-process. It arrives with the fastmcp client install, and was named
    # `FastMCP` under `mcp.server.fastmcp` before mcp 2.x — the line FastMCP 4
    # requires.
    from mcp.server.mcpserver import MCPServer
except ImportError as _import_error:
    msg = (
        "Please install the fastmcp client to use `MCPAdapter` — "
        '`pip install "fastmcp-slim[client]"`.'
    )
    raise ImportError(msg) from _import_error


if TYPE_CHECKING:
    from types import TracebackType

    from fastmcp import FastMCP
    from langchain_core.tools import BaseTool
else:
    # In-process FastMCP servers live in the server half of FastMCP, which the
    # lightweight `fastmcp-slim[client]` install does not pull in. FastMCP degrades
    # this annotation the same way in `fastmcp.client.transports.inference`.
    FastMCP = Any


MCPAdapterTarget: TypeAlias = (
    FastMCPClient[Any]
    | ClientTransport
    | FastMCP
    | MCPServer
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


async def _declare_elicitation_capability(*_: object) -> dict[str, Any]:
    """Stand in for an elicitation handler so the client advertises the capability.

    FastMCP declares `elicitation` in the client's capabilities only when it was
    given a handler — the SDK compares the callback against its own default by
    identity and declares the capability when they differ. Interrupt-driven
    elicitation answers from the tool call rather than from a callback, so this
    exists to make that declaration, and is not expected to run.

    Reaching it means a server asked for input over the legacy server-initiated
    path, which an interrupt cannot answer: FastMCP converts whatever a handler
    raises into an MCP error, so the `GraphInterrupt` would never leave here.
    """
    msg = (
        "This MCP server asked for input over the legacy server-initiated "
        "path, which `elicitation='interrupt'` cannot answer. Interrupt-based "
        "elicitation needs a server that returns its input requests as an "
        "`InputRequiredResult`."
    )
    raise NotImplementedError(msg)


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
        elicitation: Whether these tools can answer a server that needs input
            mid-call. Pass `'interrupt'` to raise a LangGraph `interrupt()` when
            a server asks, so a human answers and the run resumes — see
            `langchain.mcp.elicitation`.

            This is a promise made on the wire, which is why it is a choice
            rather than a default. A client declares up front whether it can be
            asked questions, and servers only build flows that depend on it once
            a client says yes. An agent with no way to reach a human cannot keep
            that promise, so by default nothing is declared and a server does
            not ask — which also means a server whose tool *requires* an answer
            will refuse the call outright rather than run without one.

            With a pre-built client, the declaration is that client's to make:
            it must carry an `elicitation_handler` of its own, or servers will
            not ask no matter what is passed here.

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
        elicitation: Literal["interrupt"] | None = None,
    ) -> None:
        """Initialize the adapter around a FastMCP target or client."""
        if isinstance(target, FastMCPClient):
            self._client: FastMCPClient[Any] = target
        else:
            # A client only advertises the elicitation capability when it
            # carries a handler, and servers will not ask without it. The
            # handler itself is never used: elicitation answers are driven by
            # `call_tool_with_interrupts`, which reads the requests off the
            # result rather than through this callback.
            self._client = FastMCPClient(
                target,
                elicitation_handler=(
                    _declare_elicitation_capability if elicitation == "interrupt" else None
                ),
            )
        self._elicitation = elicitation

    @property
    def client(self) -> FastMCPClient[Any]:
        """Return the underlying FastMCP client for advanced MCP operations."""
        return self._client

    async def __aenter__(self) -> Self:
        """Connect the underlying client for the duration of the context."""
        await self._client.__aenter__()  # type: ignore[no-untyped-call]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release this context's hold on the underlying client."""
        await self._client.__aexit__(exc_type, exc_value, traceback)  # type: ignore[no-untyped-call]

    async def get_tools(self) -> list[BaseTool]:
        """Discover and adapt MCP tools for use with LangChain.

        Returns:
            LangChain tools that invoke the corresponding MCP tools
                asynchronously. Each holds the adapter's client, so the tools
                stay callable after this adapter's context exits.
        """
        async with self._client:
            remote_tools = await self._client.list_tools()
        return [
            convert_mcp_tool_to_langchain_tool(tool, self._client, elicitation=self._elicitation)
            for tool in remote_tools
        ]


__all__ = ["MCPAdapter", "MCPAdapterTarget"]
