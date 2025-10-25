"""Anthropic-specific middleware for the Claude bash tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware.shell_tool import ShellToolMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

_CLAUDE_BASH_DESCRIPTOR = {"type": "bash_20250124", "name": "bash"}


class ClaudeBashToolMiddleware(ShellToolMiddleware):
    """Middleware that exposes Anthropic's native bash tool to models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize middleware without registering a client-side tool."""
        kwargs["shell_command"] = ("/bin/bash",)
        super().__init__(*args, **kwargs)
        # Remove the base tool so Claude's native descriptor is the sole entry.
        self._tool = None  # type: ignore[assignment]
        self.tools = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Ensure the Claude bash descriptor is available to the model."""
        tools = request.tools
        if all(tool is not _CLAUDE_BASH_DESCRIPTOR for tool in tools):
            tools = [*tools, _CLAUDE_BASH_DESCRIPTOR]
            request = request.override(tools=tools)
        return handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Command | ToolMessage],
    ) -> Command | ToolMessage:
        """Intercept Claude bash tool calls and execute them locally."""
        tool_call = request.tool_call
        if tool_call.get("name") != "bash":
            return handler(request)
        resources = self._ensure_resources(request.state)
        return self._run_shell_tool(
            resources,
            tool_call["args"],
            tool_call_id=tool_call.get("id"),
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Command | ToolMessage]],
    ) -> Command | ToolMessage:
        """Async interception mirroring the synchronous implementation."""
        tool_call = request.tool_call
        if tool_call.get("name") != "bash":
            return await handler(request)
        resources = self._ensure_resources(request.state)
        return self._run_shell_tool(
            resources,
            tool_call["args"],
            tool_call_id=tool_call.get("id"),
        )

    def _format_tool_message(
        self,
        content: str,
        tool_call_id: str | None,
        *,
        status: Literal["success", "error"],
        artifact: dict[str, Any] | None = None,
    ) -> ToolMessage | str:
        """Format tool responses using Claude's bash descriptor."""
        if tool_call_id is None:
            return content
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=_CLAUDE_BASH_DESCRIPTOR["name"],
            status=status,
            artifact=artifact or {},
        )


__all__ = ["ClaudeBashToolMiddleware"]
