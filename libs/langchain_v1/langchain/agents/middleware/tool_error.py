"""Tool error middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import ContentBlock
    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool

    OnError = Callable[[Exception, ToolCallRequest], str | list[ContentBlock] | None]
    """Sync handler: return content to surface the error as a `ToolMessage`; return
    `None` (or nothing) to let the exception propagate."""

    AOnError = Callable[[Exception, ToolCallRequest], Awaitable[str | list[ContentBlock] | None]]
    """Async handler: return content to surface the error as a `ToolMessage`; return
    `None` (or nothing) to let the exception propagate."""


class ToolErrorMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Return selected tool-execution exceptions to the model as error `ToolMessage`s.

    `on_error` is called for each exception raised by tool execution. Return content
    (a `str` or a list of content blocks) to convert the exception into a
    `ToolMessage(status="error")`; return `None` — or simply don't return — to let the
    exception propagate (halting the run). Handling is therefore opt-in — exceptions you
    do not return content for propagate unchanged, so arbitrary internal exceptions are
    never serialized to the model or end user unless you choose to surface them.

    Langgraph control-flow signals (interrupts, parent commands) always propagate and
    never reach `on_error`.

    Prefer returning content that names the exception type over the raw exception message,
    which may carry sensitive or internal detail.

    Provide at least one of `on_error` or `aon_error`. `aon_error` handles errors on the
    async execution path (falling back to `on_error` when omitted); the sync path only
    ever calls `on_error`. For async-only usage, pass `aon_error` alone — running such a
    middleware on the sync path raises, since the async handler cannot be awaited there.

    This middleware does not retry. For retries, compose with `ToolRetryMiddleware`
    placed *inner* and configured with `on_failure="error"` so exceptions reach this
    middleware.

    This middleware only sees exceptions raised by tool *execution*. Argument-binding
    and validation errors are handled upstream by `ToolNode` (converted to an error
    `ToolMessage` before the tool runs), so they do not reach `on_error`.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ToolErrorMiddleware


        def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
            if isinstance(exc, ValueError):
                return f"`{request.tool_call['name']}` failed; fix the input and retry."
            return None  # propagate everything else


        agent = create_agent(model, tools=[...], middleware=[ToolErrorMiddleware(on_error)])
        ```
    """

    def __init__(
        self,
        on_error: OnError | None = None,
        *,
        aon_error: AOnError | None = None,
        tools: list[BaseTool | str] | None = None,
    ) -> None:
        """Initialize `ToolErrorMiddleware`.

        Args:
            on_error: Handler called for each exception raised by tool execution. Return
                content (`str` or list of content blocks) to convert the exception into an
                error `ToolMessage`. Return `None` — or simply don't return — to let the
                exception propagate. Falling through without a return therefore re-raises,
                so handle only the exceptions you mean to. Receives the exception and the
                tool call request (tool name, args, call id). Used on the sync path and,
                unless `aon_error` is given, on the async path.
            aon_error: Optional async handler, used on the async execution path. Falls back
                to `on_error` when not provided.
            tools: Optional list of tools or tool names to apply handling to. If `None`,
                applies to all tools.

        Raises:
            ValueError: If neither `on_error` nor `aon_error` is provided.
        """
        super().__init__()

        if on_error is None and aon_error is None:
            msg = "ToolErrorMiddleware requires `on_error` and/or `aon_error`."
            raise ValueError(msg)

        self.on_error = on_error
        self.aon_error = aon_error

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

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool execution and convert handled exceptions to error messages.

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
            if self.on_error is None:
                # Async-only config (aon_error) cannot be awaited on the sync path.
                msg = (
                    "ToolErrorMiddleware has no sync `on_error`; run async "
                    "(ainvoke/astream) or provide `on_error`."
                )
                raise RuntimeError(msg) from exc
            content = self.on_error(exc, request)
            if content is None:
                raise
            return ToolMessage(
                content=cast("str | list[str | dict[Any, Any]]", content),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of `wrap_tool_call`.

        Uses `aon_error` if provided, otherwise the sync `on_error`. The sync path never
        awaits.
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        if not self._should_handle_tool(tool_name):
            return await handler(request)

        try:
            return await handler(request)
        except GraphBubbleUp:
            # Control-flow signals (interrupts, parent commands) must propagate.
            raise
        except Exception as exc:
            if self.aon_error is not None:
                content = await self.aon_error(exc, request)
            elif self.on_error is not None:
                content = self.on_error(exc, request)
            else:  # pragma: no cover - __init__ guarantees at least one handler
                raise
            if content is None:
                raise
            return ToolMessage(
                content=cast("str | list[str | dict[Any, Any]]", content),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )
