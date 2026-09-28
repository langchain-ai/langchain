"""Adapt MCP tools into LangChain tools suitable for `create_agent`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias

from pydantic import AnyUrl, TypeAdapter, ValidationError
from typing_extensions import Self

from langchain.mcp.elicitation import _arm_for_interrupts
from langchain.mcp.tools import as_langchain_tool

try:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.group import ClientGroup
    from fastmcp.client.transports import ClientTransport
    from fastmcp.mcp_config import MCPConfig
    from mcp.server.mcpserver import MCPServer
except ImportError as _import_error:
    msg = "Please install FastMCP to use `MCPAdapter` — `pip install fastmcp`."
    raise ImportError(msg) from _import_error


if TYPE_CHECKING:
    from types import TracebackType

    from fastmcp import FastMCP
    from langchain_core.tools import BaseTool

    # `CacheMode` is the MCP SDK's own type (SEP-2549); FastMCP re-imports it
    # rather than defining its own, so this is the canonical source.
    from mcp.client.caching import CacheMode
else:
    try:
        from fastmcp import FastMCP
    except ImportError:
        # In-process FastMCP servers, and multi-server `MCPConfig` targets, live
        # in the server half of FastMCP. The `mcp` extra installs it, but a
        # caller who reached this module through a client-only install
        # (`fastmcp-slim[client]`) can still use every other target type — so
        # the annotation degrades rather than the import failing, the same way
        # FastMCP degrades it in `fastmcp.client.transports.inference`.
        FastMCP = Any


MCPAdapterTarget: TypeAlias = (
    FastMCPClient[Any]
    | ClientGroup
    | ClientTransport
    | FastMCP
    | MCPServer
    | AnyUrl
    | Path
    | MCPConfig
    | dict[str, Any]
    | str
)
"""Anything `MCPAdapter` accepts as its `target`.

Every transport `fastmcp.Client` accepts, plus a pre-built `fastmcp.Client`.

A `str` target is the one member narrowed relative to `fastmcp.Client`: it must
be an `http`/`https` URL. Local servers are reached through `Path`, an explicit
transport, or `MCPConfig` — see `MCPAdapter`.

`ClientGroup` is the one member `fastmcp.Client` does not accept at all, since a
group is a peer of `Client` rather than a transport it could wrap.
"""


_URL_SCHEMES: Final = frozenset({"http", "https"})
"""Schemes a `str` target may carry.

The only schemes `fastmcp.Client` can reach a server over from a string, so
nothing that would have connected is forbidden here. It selects that branch on
a bare `str.startswith("http")` though, which also admits strings that are not
URLs at all — those fail on connect there, and on construction here.
"""

_ANY_URL: Final = TypeAdapter(AnyUrl)


def _validate_url_target(target: str) -> None:
    """Reject a `str` target that is not an `http`/`https` URL.

    `fastmcp.Client` infers a transport from a string by testing it as a
    filesystem path *before* testing it as a URL, so a string naming an
    existing `.py` or `.js` file launches that file as a subprocess. A string
    that reaches an application from configuration, a request body, or a model
    is expected to address a server over the network; letting it silently
    select local execution instead makes the dangerous reading the implicit
    one.

    Args:
        target: The string passed as the adapter's target.

    Raises:
        ValueError: If `target` is not an `http` or `https` URL.
    """
    hint = (
        "To run a local MCP server over stdio, ask for it explicitly with "
        "`Path('server.py')`, a `fastmcp` transport, or an `MCPConfig` — a bare "
        "string is read as a URL so that it cannot silently become a subprocess."
    )
    try:
        url = _ANY_URL.validate_python(target)
    except ValidationError as validation_error:
        msg = f"MCP target {target!r} is not a valid URL. {hint}"
        raise ValueError(msg) from validation_error

    # `AnyUrl` alone is not the boundary. It reads a Windows path such as
    # `C:\server.py` as the scheme `c` and accepts it — and on Windows that
    # path *does* resolve, so FastMCP would launch it. Schemes FastMCP cannot
    # infer a transport from at all (`file:`, `ws:`) are refused here too, for
    # an error that names the problem rather than "could not infer a transport".
    if url.scheme not in _URL_SCHEMES:
        msg = (
            f"MCP target {target!r} has scheme {url.scheme!r}, but a string target must "
            f"be an http or https URL. {hint}"
        )
        raise ValueError(msg)


class MCPAdapter:
    """Adapt an MCP target into LangChain tools.

    `MCPAdapter` uses FastMCP for protocol negotiation and connection management,
    then converts discovered MCP tools into asynchronous LangChain tools. The
    resulting tools can be passed directly to `create_agent`.

    Transport inference is delegated to `fastmcp.Client`, so a target may be a URL,
    a local script path (launched over stdio), an in-process server, or an already
    constructed client.

    !!! warning "A string target must be an http(s) URL"

        `fastmcp.Client` resolves a string by testing it as a filesystem path
        before testing it as a URL, so a string naming an existing `.py` or `.js`
        file launches that file as a subprocess. Because strings are the form a
        target most often arrives in from configuration or from a model,
        `MCPAdapter` rejects one that is not an `http` or `https` URL rather than
        let it select local execution. Reach a local server through `Path`, a
        `fastmcp` transport, or an `MCPConfig`, all of which say so explicitly.

    A server that needs input mid-call is answered with a LangGraph
    `interrupt()`, so a human answers and the run resumes — see
    `langchain.mcp.elicitation`. This is the default: the adapter arms every
    client it builds to advertise the elicitation capability and drives the
    interrupt loop on each call. A server that never asks for input is
    unaffected, since the loop only runs when the server returns a request.

    A pre-built client (or `ClientGroup`) that already carries its own
    elicitation handler is honored rather than overridden: its handler keeps
    answering, and the adapter leaves the client as the caller built it. Only a
    client with no handler is armed, and it is cloned first so the caller's own
    object is never mutated.

    Args:
        target: MCP target accepted by `fastmcp.Client`, including an existing
            FastMCP client. A `str` must be an `http`/`https` URL.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.mcp import MCPAdapter

        async with MCPAdapter("https://example.com/mcp") as adapter:
            agent = create_agent("anthropic:claude-sonnet-5", await adapter.list_tools())
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
        ```
    """

    def __init__(self, target: MCPAdapterTarget) -> None:
        """Initialize the adapter around a FastMCP target, client, or group.

        Each underlying client is armed to answer elicitation with an interrupt,
        unless it already carries the caller's own handler, which is honored
        instead. A caller's client is cloned rather than mutated. See the class
        docstring.
        """

        def armed(client: FastMCPClient[Any]) -> FastMCPClient[Any]:
            if getattr(client, "_elicitation_callback", None) is not None:
                return client
            clone = client.new()
            _arm_for_interrupts(clone)
            return clone

        if isinstance(target, ClientGroup):
            members = {name: armed(client) for name, client in target.clients.items()}
            self._client: FastMCPClient[Any] | ClientGroup = ClientGroup(members)
        elif isinstance(target, FastMCPClient):
            self._client = armed(target)
        else:
            if isinstance(target, str):
                _validate_url_target(target)
            client = FastMCPClient(target)
            _arm_for_interrupts(client)
            self._client = client

    @property
    def client(self) -> FastMCPClient[Any] | ClientGroup:
        """Return the underlying FastMCP client or group for advanced MCP operations."""
        return self._client

    async def __aenter__(self) -> Self:
        """Connect the underlying client or group for the duration of the context."""
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release this context's hold on the underlying client or group."""
        await self._client.__aexit__(exc_type, exc_value, traceback)

    async def list_tools(self, *, cache_mode: CacheMode = "use") -> list[BaseTool]:
        """Discover and adapt MCP tools for use with LangChain.

        Args:
            cache_mode: How discovery interacts with the client-side response
                cache (SEP-2549). `use` serves a cached tool list when one is
                present and still within the server's TTL hint, `refresh` calls
                the server and repopulates the cache, and `bypass` skips the
                cache entirely. The cache and its per-principal isolation are
                configured on the client itself (`Client(cache=...)`); this only
                selects how discovery reads it. Defaults to `use` so a
                configured cache is honored — note this differs from a bare
                `ClientGroup.list_tools()`, whose own default is `refresh`.

        Returns:
            LangChain tools that invoke the corresponding MCP tools
                asynchronously. Each holds the adapter's client, so the tools
                stay callable after this adapter's context exits.
        """
        async with self:
            remote_tools = await self._client.list_tools(cache_mode=cache_mode)
            return [await as_langchain_tool(tool, self._client) for tool in remote_tools]


__all__ = ["MCPAdapter", "MCPAdapterTarget"]
