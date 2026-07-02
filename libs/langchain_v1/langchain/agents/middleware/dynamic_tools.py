"""Middleware helpers for dynamic tool registration at runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from typing_extensions import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ResponseT,
    ToolCallRequest,
)


class DynamicToolsMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Register additional tools for each model call via middleware.

    Tools passed to this middleware are merged into the model request on every
    turn. Pair with `wrap_tool_call` when tools are not known at agent
    construction time.
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        """Initialize with tools to inject on each model call."""
        super().__init__()
        self._dynamic_tools = list(tools)

    @property
    @override
    def tools(self) -> list[BaseTool]:
        return self._dynamic_tools

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelCallResult[ResponseT]:
        updated = request.override(tools=[*request.tools, *self._dynamic_tools])
        return handler(updated)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelCallResult[ResponseT]:
        updated = request.override(tools=[*request.tools, *self._dynamic_tools])
        return await handler(updated)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        for tool in self._dynamic_tools:
            if tool.name == request.tool_call["name"]:
                return handler(request.override(tool=tool))
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        for tool in self._dynamic_tools:
            if tool.name == request.tool_call["name"]:
                return await handler(request.override(tool=tool))
        return await handler(request)
