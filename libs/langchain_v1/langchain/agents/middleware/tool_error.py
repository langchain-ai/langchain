"""Tool error middleware for agents."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool

Catch = tuple[type[Exception], ...] | Callable[[Exception], bool]
"""Exceptions to catch: a tuple of exception types, or a predicate ``(exc) -> bool``."""

OnError = Callable[
    [Exception, "ToolCallRequest"],
    "str | list[str | dict[Any, Any]] | Awaitable[str | list[str | dict[Any, Any]]]",
]
"""Handler for a caught exception, returning `ToolMessage` content.

May be sync or async. An async `on_error` requires async execution (`ainvoke`/`astream`).
"""


def _should_catch(exc: Exception, catch: Catch) -> bool:
    """Return whether `exc` should be caught and returned to the model."""
    if isinstance(catch, tuple):
        return isinstance(exc, catch)
    return bool(catch(exc))


class ToolErrorMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Return tool-execution exceptions to the model as error `ToolMessage`s.

    Only the exceptions named in `catch` are converted into a
    `ToolMessage(status="error")`; any other exception propagates and halts the run.
    Langgraph control-flow signals (interrupts, parent commands) always propagate.

    `catch` is required: there is intentionally no catch-all default, so arbitrary
    internal exceptions are not serialized to the model or end user. Use `on_error`
    to control (and sanitize) exactly what content the model sees.

    This middleware does not retry. For retries, compose with `ToolRetryMiddleware`
    placed *inner* and configured with `on_failure="error"` so exceptions reach this
    middleware.

    This middleware only sees exceptions raised by tool *execution*. Argument-binding
    and validation errors are handled upstream by `ToolNode` (converted to an error
    `ToolMessage` before the tool runs), so they do not pass through `catch` or
    `on_error` and are not sanitized by `on_error`.

    Guidance on what to `catch`:

    - **Catch** (return to the model): anticipated, model-actionable, non-sensitive
        errors raised by the tool — e.g. tool-domain errors the model can correct.
    - **Do not catch** (let propagate): programming bugs, auth/permission errors, and
        anything whose message may carry secrets or internal infrastructure detail.

    Examples:
        !!! example "Catch a specific tool error"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ToolErrorMiddleware

            agent = create_agent(
                model,
                tools=[search_tool],
                middleware=[ToolErrorMiddleware(catch=(ValueError,))],
            )
            ```

        !!! example "Custom, sanitized error message"

            ```python
            def on_error(exc: Exception, request: ToolCallRequest) -> str:
                name = request.tool_call["name"]
                return f"`{name}` failed with invalid input. Check the arguments and retry."


            ToolErrorMiddleware(catch=(ValueError,), on_error=on_error)
            ```
    """

    def __init__(
        self,
        catch: Catch,
        *,
        on_error: OnError | None = None,
        tools: list[BaseTool | str] | None = None,
    ) -> None:
        """Initialize `ToolErrorMiddleware`.

        Args:
            catch: Exceptions to convert into an error `ToolMessage`. Either a tuple of
                exception types, or a predicate `(exc) -> bool`. Exceptions that are not
                caught propagate (halting the run).
            on_error: Optional formatter for the `ToolMessage` content. Receives the
                exception and the tool call request (tool name, args, call id). Defaults
                to a conservative prose message. May return a string or a list of
                content blocks.
            tools: Optional list of tools or tool names to apply handling to. Can be a
                list of `BaseTool` instances or tool name strings. If `None`, applies to
                all tools.
        """
        super().__init__()

        self.catch = catch
        self.on_error = on_error

        # Extract tool names from BaseTool instances or strings
        self._tool_filter: list[str] | None
        if tools is not None:
            self._tool_filter = [tool.name if not isinstance(tool, str) else tool for tool in tools]
        else:
            self._tool_filter = None

        self.tools = []  # No additional tools registered by this middleware

    def _should_handle_tool(self, tool_name: str) -> bool:
        """Check if error handling should apply to this tool."""
        if self._tool_filter is None:
            return True
        return tool_name in self._tool_filter

    @staticmethod
    def _format_error(tool_name: str, exc: Exception) -> str:
        """Default formatter for a caught exception.

        Names the exception type but omits its message, which may contain sensitive
        or internal detail. Provide `on_error` to include a sanitized message.
        """
        return f"Tool '{tool_name}' failed with {type(exc).__name__}."

    def _sync_content(
        self, exc: Exception, request: ToolCallRequest, tool_name: str
    ) -> str | list[str | dict[Any, Any]]:
        """Resolve the error message content in a sync context."""
        if self.on_error is None:
            return self._format_error(tool_name, exc)
        result = self.on_error(exc, request)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            msg = "async on_error requires async execution (ainvoke or astream)"
            raise TypeError(msg)
        return result

    async def _async_content(
        self, exc: Exception, request: ToolCallRequest, tool_name: str
    ) -> str | list[str | dict[Any, Any]]:
        """Resolve the error message content in an async context."""
        if self.on_error is None:
            return self._format_error(tool_name, exc)
        result = self.on_error(exc, request)
        if inspect.isawaitable(result):
            return await result
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool execution and convert caught exceptions to error messages.

        Args:
            request: Tool call request with call dict, `BaseTool`, state, and runtime.
            handler: Callable to execute the tool.

        Returns:
            `ToolMessage` or `Command` (the final result).
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        if not self._should_handle_tool(tool_name):
            return handler(request)

        try:
            return handler(request)
        except GraphBubbleUp:
            # Control-flow signals (interrupts, parent commands) must propagate.
            raise
        except Exception as exc:
            if not _should_catch(exc, self.catch):
                raise
            return ToolMessage(
                content=self._sync_content(exc, request, tool_name),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of `wrap_tool_call`."""
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        if not self._should_handle_tool(tool_name):
            return await handler(request)

        try:
            return await handler(request)
        except GraphBubbleUp:
            # Control-flow signals (interrupts, parent commands) must propagate.
            raise
        except Exception as exc:
            if not _should_catch(exc, self.catch):
                raise
            return ToolMessage(
                content=await self._async_content(exc, request, tool_name),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )
