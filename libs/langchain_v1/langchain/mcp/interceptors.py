"""Interceptor interfaces and types for MCP client tool call lifecycle management.

This module provides an interceptor interface for wrapping and controlling
MCP tool call execution with a handler callback pattern.

In the future, we might add more interceptors for other parts of the
request / result lifecycle, for example to support elicitation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from langchain_core.messages import ToolMessage
from mcp.types import CallToolResult
from typing_extensions import NotRequired, TypedDict, Unpack

try:
    # langgraph installed
    import langgraph  # noqa: F401

    LANGGRAPH_PRESENT = True
except ImportError:
    LANGGRAPH_PRESENT = False


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

if LANGGRAPH_PRESENT:
    from langgraph.types import Command

    MCPToolCallResult = CallToolResult | ToolMessage | Command
else:
    MCPToolCallResult = CallToolResult | ToolMessage


class _MCPToolCallRequestOverrides(TypedDict, total=False):
    """Possible overrides for MCPToolCallRequest.override() method.

    Only includes modifiable request fields, not context fields like
    server_name and runtime which are read-only.
    """

    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    headers: NotRequired[dict[str, Any] | None]


@dataclass
class MCPToolCallRequest:
    """Tool execution request passed to MCP tool call interceptors.

    This tool call request follows a similar pattern to LangChain's
    ToolCallRequest (flat namespace) rather than separating the call data
    and context into nested objects.

    Modifiable fields (override these to change behavior):
        name: Tool name to invoke.
        args: Tool arguments as key-value pairs.
        headers: HTTP headers for applicable transports (SSE, HTTP).

    Context fields (read-only, use for routing/logging):
        server_name: Name of the MCP server handling the tool.
        runtime: LangGraph runtime context (optional, None if outside graph).
    """

    name: str
    args: dict[str, Any]
    server_name: str  # Context: MCP server name
    headers: dict[str, Any] | None = None  # Modifiable: HTTP headers
    runtime: object | None = None  # Context: LangGraph runtime (if any)

    def override(self, **overrides: Unpack[_MCPToolCallRequestOverrides]) -> MCPToolCallRequest:
        """Replace the request with a new request with the given overrides.

        Returns a new `MCPToolCallRequest` instance with the specified
        attributes replaced. This follows an immutable pattern, leaving the
        original request unchanged.

        Args:
            **overrides: Keyword arguments for attributes to override.
                Supported keys:
                - name: Tool name
                - args: Tool arguments
                - headers: HTTP headers

        Returns:
            New MCPToolCallRequest instance with specified overrides
            applied.

        Note:
            Context fields (server_name, runtime) cannot be overridden as
            they are read-only.

        Examples:
            ```python
            # Modify tool arguments
            new_request = request.override(args={"value": 10})

            # Change tool name
            new_request = request.override(name="different_tool")
            ```
        """
        return replace(self, **overrides)


@runtime_checkable
class ToolCallInterceptor(Protocol):
    """Protocol for tool call interceptors using handler callback pattern.

    Interceptors wrap tool execution to enable request/response modification,
    retry logic, caching, rate limiting, and other cross-cutting concerns.
    Multiple interceptors compose in "onion" pattern (first is outermost).

    The handler can be called multiple times (retry), skipped (caching/short-circuit),
    or wrapped with error handling. Each handler call is independent.

    Similar to LangChain's middleware pattern but adapted for MCP remote tools.
    """

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        """Intercept tool execution with control over handler invocation.

        Args:
            request: Tool call request containing name, args, headers, and context
                (server_name, runtime). Access context fields like request.server_name.
            handler: Async callable executing the tool. Can be called multiple
                times, skipped, or wrapped for error handling.

        Returns:
            Final MCPToolCallResult from tool execution or interceptor logic.
        """
        ...
