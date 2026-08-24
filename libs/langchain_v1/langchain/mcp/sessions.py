"""Connection configuration and session management for MCP servers.

Connections are described with the MCP SDK's own types and opened with `create_session`,
which builds an [`mcp.Client`][mcp.Client] and yields its session. Transports, protocol
negotiation, and HTTP client construction all belong to that client.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeAlias

from mcp import Client, StdioServerParameters
from mcp.client.session_group import (
    ServerParameters,
    SseServerParameters,
    StreamableHttpParameters,
)
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp import ClientSession

    from langchain.mcp.callbacks import _MCPCallbacks

Connection: TypeAlias = ServerParameters | Mapping[str, Any] | Callable[[], Any]
"""How to reach one MCP server.

One of:

- The MCP SDK's [`ServerParameters`][mcp.client.session_group.ServerParameters] —
  `StdioServerParameters`, `SseServerParameters`, or `StreamableHttpParameters`.
- A mapping of the same fields, selected by an optional `transport` key of `"stdio"`,
  `"sse"`, or `"streamable_http"` and otherwise inferred from `command` or `url`. The
  mapping additionally accepts `auth` and `http_client` for the HTTP transports, and any
  [`Client`][mcp.Client] option such as `mode`, none of which the SDK's parameter models
  describe.
- A callable returning a transport or a `Client`, for anything the above cannot express.
  It must be a callable rather than an instance, because a session is opened per
  operation and both are single-use:

  ```python
  lambda: streamable_http_client(url, http_client=authed_client)
  lambda: Client(transport, mode="legacy", cache=cache_config)
  ```
"""

_PARAMETERS_BY_TRANSPORT: dict[str, type[ServerParameters]] = {
    "stdio": StdioServerParameters,
    "sse": SseServerParameters,
    "http": StreamableHttpParameters,
    "streamable_http": StreamableHttpParameters,
    "streamable-http": StreamableHttpParameters,
}

# Keys that configure the transport or the client rather than describing the server. The
# SDK's parameter models ignore keys they do not declare, so these ride along in a mapping.
_TRANSPORT_ONLY_KEYS = frozenset({"auth", "http_client"})
_NOT_CLIENT_KEYS = frozenset({"transport", "session_kwargs"}) | _TRANSPORT_ONLY_KEYS


def _transport_from_parameters(
    params: ServerParameters,
    *,
    auth: Any = None,
    http_client: Any = None,
) -> Any:
    """Build a transport from the SDK's connection parameters.

    `auth` and `http_client` are taken separately because the SDK's HTTP parameter models
    describe neither, and they are the usual reason to reach for a transport.

    Raises:
        ValueError: If the parameters are of an unrecognized type.
    """
    if isinstance(params, StdioServerParameters):
        return stdio_client(params)
    if isinstance(params, SseServerParameters):
        return sse_client(
            params.url,
            headers=params.headers,
            timeout=params.timeout,
            sse_read_timeout=params.sse_read_timeout,
            auth=auth,
        )
    if isinstance(params, StreamableHttpParameters):
        return streamable_http_client(
            params.url,
            http_client=http_client or create_mcp_http_client(headers=params.headers, auth=auth),
            terminate_on_close=params.terminate_on_close,
        )
    msg = f"Unsupported server parameters: {type(params).__name__}"
    raise ValueError(msg)


def _parse_mapping(config: Mapping[str, Any]) -> tuple[ServerParameters, dict[str, Any]]:
    """Split a mapping into the SDK's server parameters and the client options around them.

    Raises:
        ValueError: If the transport is unknown.
    """
    name = config.get("transport") or ("stdio" if "command" in config else "http")
    model = _PARAMETERS_BY_TRANSPORT.get(name)
    if model is None:
        known = ", ".join(sorted(_PARAMETERS_BY_TRANSPORT))
        msg = f"Unsupported transport: {name}. Must be one of: {known}"
        raise ValueError(msg)

    params = model.model_validate(config)
    client_kwargs = {
        key: value
        for key, value in config.items()
        if key not in _NOT_CLIENT_KEYS and key not in model.model_fields
    }
    client_kwargs.update(config.get("session_kwargs") or {})
    return params, client_kwargs


def _callback_kwargs(mcp_callbacks: _MCPCallbacks | None) -> dict[str, Any]:
    """Client options for the callbacks the caller registered, if any."""
    if mcp_callbacks is None:
        return {}
    return {
        name: callback
        for name, callback in (
            ("logging_callback", mcp_callbacks.logging_callback),
            ("elicitation_callback", mcp_callbacks.elicitation_callback),
        )
        if callback is not None
    }


@asynccontextmanager
async def create_session(
    connection: Connection, *, mcp_callbacks: _MCPCallbacks | None = None
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server.

    Args:
        connection: How to reach the server. See `Connection`.
        mcp_callbacks: mcp sdk compatible callbacks to use for the session

    Raises:
        TypeError: If a transport or client instance is passed instead of a callable
            returning one, or the connection is of an unusable type.
        ValueError: If the transport is unknown.

    Yields:
        A ClientSession
    """
    callbacks = _callback_kwargs(mcp_callbacks)

    if isinstance(connection, ServerParameters):
        client = Client(_transport_from_parameters(connection), **callbacks)
    elif isinstance(connection, Mapping):
        params, options = _parse_mapping(connection)
        transport = _transport_from_parameters(
            params,
            auth=connection.get("auth"),
            http_client=connection.get("http_client"),
        )
        client = Client(transport, **{**options, **callbacks})
    elif hasattr(connection, "__aenter__"):
        # Checked before the callable case, because transports and clients are context
        # managers that also happen to be callable.
        msg = (
            "A transport or client cannot be used as a connection, because it can only "
            "be entered once and a session is opened per operation. Pass a callable "
            "returning a fresh one instead, such as "
            "`lambda: streamable_http_client(url, http_client=...)`."
        )
        raise TypeError(msg)
    elif callable(connection):
        built = connection()
        client = built if isinstance(built, Client) else Client(built, **callbacks)
    else:
        msg = (
            "Unsupported connection. Expected the MCP SDK's ServerParameters, a mapping "
            "of the same fields, or a callable returning a transport or client; got "
            f"{type(connection).__name__}."
        )
        raise TypeError(msg)

    async with client:
        yield client.session
