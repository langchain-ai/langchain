"""Session management for different MCP transport types.

This module provides connection configurations and session management for various
MCP transport types including stdio, SSE, and streamable HTTP.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


EncodingErrorHandler = Literal["strict", "ignore", "replace"]

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = "strict"

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5

DEFAULT_STREAMABLE_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT = timedelta(seconds=60 * 5)


class McpHttpClientFactory(Protocol):
    """Protocol for creating httpx2.AsyncClient instances for MCP connections."""

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient:
        """Create an httpx2.AsyncClient instance.

        Args:
            headers: HTTP headers to include in requests.
            timeout: Request timeout configuration.
            auth: Authentication configuration.

        Returns:
            Configured httpx2.AsyncClient instance.
        """
        ...


class StdioConnection(TypedDict):
    """Configuration for stdio transport connections to MCP servers."""

    transport: Literal["stdio"]

    command: str
    """The executable to run to start the server."""

    args: list[str]
    """Command line arguments to pass to the executable."""

    env: NotRequired[dict[str, str] | None]
    """The environment to use when spawning the process.

    If not specified or set to None, a subset of the default environment
    variables from the current process will be used.

    Please refer to the MCP SDK documentation for details on which
    environment variables are included by default. The behavior
    varies by operating system.

    https://github.com/modelcontextprotocol/python-sdk/blob/c47c767ff437ee88a19e6b9001e2472cb6f7d5ed/src/mcp/client/stdio/__init__.py#L51
    """

    cwd: NotRequired[str | Path | None]
    """The working directory to use when spawning the process."""

    encoding: NotRequired[str]
    """The text encoding used when sending/receiving messages to the server.

    Default is 'utf-8'.
    """

    encoding_error_handler: NotRequired[EncodingErrorHandler]
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.

    Default is 'strict', which raises an error on encoding/decoding errors.
    """

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession."""


class SSEConnection(TypedDict):
    """Configuration for Server-Sent Events (SSE) transport connections to MCP."""

    transport: Literal["sse"]

    url: str
    """The URL of the SSE endpoint to connect to."""

    headers: NotRequired[dict[str, Any] | None]
    """HTTP headers to send to the SSE endpoint."""

    timeout: NotRequired[float]
    """HTTP timeout.

    Default is 5 seconds. If the server takes longer to respond,
    you can increase this value.
    """

    sse_read_timeout: NotRequired[float]
    """SSE read timeout.

    Default is 300 seconds (5 minutes). This is how long the client will
    wait for a new event before disconnecting.
    """

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession."""

    httpx_client_factory: NotRequired[McpHttpClientFactory | None]
    """Custom factory for httpx2.AsyncClient (optional)."""

    auth: NotRequired[httpx2.Auth]
    """Optional authentication for the HTTP client."""


class StreamableHttpConnection(TypedDict):
    """Connection configuration for Streamable HTTP transport."""

    transport: Literal["streamable_http"]

    url: str
    """The URL of the endpoint to connect to."""

    headers: NotRequired[dict[str, Any] | None]
    """HTTP headers to send to the endpoint."""

    timeout: NotRequired[timedelta]
    """HTTP timeout."""

    sse_read_timeout: NotRequired[timedelta]
    """How long (in seconds) the client will wait for a new event before disconnecting.
    All other HTTP operations are controlled by `timeout`."""

    terminate_on_close: NotRequired[bool]
    """Whether to terminate the session on close."""

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession."""

    httpx_client_factory: NotRequired[McpHttpClientFactory | None]
    """Custom factory for httpx2.AsyncClient (optional)."""

    auth: NotRequired[httpx2.Auth]
    """Optional authentication for the HTTP client."""


Connection = StdioConnection | SSEConnection | StreamableHttpConnection


@asynccontextmanager
async def _create_stdio_session(
    *,
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    encoding: str = DEFAULT_ENCODING,
    encoding_error_handler: Literal["strict", "ignore", "replace"] = DEFAULT_ENCODING_ERROR_HANDLER,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using stdio.

    Args:
        command: Command to execute.
        args: Arguments for the command.
        env: Environment variables for the command.
            If not specified, inherits a subset of the current environment.
            The details are implemented in the MCP sdk.
        cwd: Working directory for the command.
        encoding: Character encoding.
        encoding_error_handler: How to handle encoding errors.
        session_kwargs: Additional keyword arguments to pass to the ClientSession.

    Yields:
        An initialized ClientSession.
    """
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        encoding=encoding,
        encoding_error_handler=encoding_error_handler,
    )

    # Create and store the connection
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write, **(session_kwargs or {})) as session,
    ):
        yield session


@asynccontextmanager
async def _create_sse_session(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
    session_kwargs: dict[str, Any] | None = None,
    httpx_client_factory: McpHttpClientFactory | None = None,
    auth: httpx2.Auth | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using SSE.

    Args:
        url: URL of the SSE server.
        headers: HTTP headers to send to the SSE endpoint.
        timeout: HTTP timeout.
        sse_read_timeout: SSE read timeout.
        session_kwargs: Additional keyword arguments to pass to the ClientSession.
        httpx_client_factory: Custom factory for httpx2.AsyncClient (optional).
        auth: Authentication for the HTTP client.

    Yields:
        An initialized ClientSession.
    """
    # Create and store the connection
    kwargs: dict[str, Any] = {}
    if httpx_client_factory is not None:
        kwargs["httpx_client_factory"] = httpx_client_factory

    async with (
        sse_client(url, headers, timeout, sse_read_timeout, auth=auth, **kwargs) as (
            read,
            write,
        ),
        ClientSession(read, write, **(session_kwargs or {})) as session,
    ):
        yield session


@asynccontextmanager
async def _create_streamable_http_session(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: timedelta = DEFAULT_STREAMABLE_HTTP_TIMEOUT,
    sse_read_timeout: timedelta = DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT,
    terminate_on_close: bool = True,
    session_kwargs: dict[str, Any] | None = None,
    httpx_client_factory: McpHttpClientFactory | None = None,
    auth: httpx2.Auth | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using Streamable HTTP.

    Args:
        url: URL of the endpoint to connect to.
        headers: HTTP headers to send to the endpoint.
        timeout: HTTP timeout.
        sse_read_timeout: How long the client will wait for a new event before
            disconnecting.
        terminate_on_close: Whether to terminate the session on close.
        session_kwargs: Additional keyword arguments to pass to the ClientSession.
        httpx_client_factory: Custom factory for httpx2.AsyncClient (optional).
        auth: Authentication for the HTTP client.

    Yields:
        An initialized ClientSession.
    """
    # `streamable_http_client` no longer takes the HTTP options directly: they belong to
    # the client it is handed, which `httpx_client_factory` may supply instead.
    factory = httpx_client_factory or create_mcp_http_client
    http_client = factory(
        headers=headers,
        timeout=httpx2.Timeout(timeout.total_seconds(), read=sse_read_timeout.total_seconds()),
        auth=auth,
    )

    async with (
        http_client,
        streamable_http_client(
            url,
            http_client=http_client,
            terminate_on_close=terminate_on_close,
        ) as (read, write),
        ClientSession(read, write, **(session_kwargs or {})) as session,
    ):
        yield session


@asynccontextmanager
async def create_session(
    connection: Connection,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server.

    Args:
        connection: Connection config to use to connect to the server

    Raises:
        ValueError: If transport is not recognized
        ValueError: If required parameters for the specified transport are missing

    Yields:
        A ClientSession
    """
    if "transport" not in connection:
        msg = (  # type: ignore[unreachable]
            "Configuration error: Missing 'transport' key in server configuration. "
            "Each server must include 'transport' with one of: "
            "'stdio', 'sse', 'http'. "
            "Please refer to the langchain-mcp-adapters documentation for more details."
        )
        raise ValueError(msg)

    transport = connection["transport"]
    params: dict[str, Any] = {k: v for k, v in connection.items() if k != "transport"}

    if transport == "sse":
        if "url" not in params:
            msg = "'url' parameter is required for SSE connection"
            raise ValueError(msg)
        async with _create_sse_session(**params) as session:
            yield session
    elif transport in {"streamable_http", "streamable-http", "http"}:
        if "url" not in params:
            msg = "'url' parameter is required for Streamable HTTP connection"
            raise ValueError(msg)
        async with _create_streamable_http_session(**params) as session:
            yield session
    elif transport == "stdio":
        if "command" not in params:
            msg = "'command' parameter is required for stdio connection"
            raise ValueError(msg)
        if "args" not in params:
            msg = "'args' parameter is required for stdio connection"
            raise ValueError(msg)
        async with _create_stdio_session(**params) as session:
            yield session
    else:
        msg = f"Unsupported transport: {transport}. Must be one of: 'stdio', 'sse', 'http'"
        raise ValueError(msg)
