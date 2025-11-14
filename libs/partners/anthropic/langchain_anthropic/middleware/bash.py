"""Anthropic-specific middleware for the Claude bash tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware.shell_tool import ShellToolMiddleware, ShellToolState
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage

_CLAUDE_BASH_DESCRIPTOR = {"type": "bash_20250124", "name": "bash"}


class ClaudeBashToolMiddleware(ShellToolMiddleware):
    """Middleware that exposes Anthropic's native bash tool to models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize middleware without registering a client-side tool."""
        kwargs["shell_command"] = ("/bin/bash",)
        super().__init__(*args, **kwargs)
        # Remove the base tool so Claude's native descriptor is the sole entry.
        self._tool = None  # type: ignore[assignment]

        # Create tool that will be executed by the tool node
        @tool("bash")
        def bash_tool(
            *,
            runtime: ToolRuntime[None, ShellToolState],
            command: str,
            restart: bool = False,
        ) -> ToolMessage | str:
            """Execute bash commands.

            Args:
                runtime: Tool runtime providing access to state and tool_call_id.
                command: The bash command to execute.
                restart: Whether to restart the shell session.

            Returns:
                The command output as ToolMessage or string.
            """
            resources = self._ensure_resources(runtime.state)
            return self._run_shell_tool(
                resources,
                {"command": command, "restart": restart},
                tool_call_id=runtime.tool_call_id,
            )

        self.tools = [bash_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Replace our tool with Claude's bash descriptor."""
        # Filter out our registered bash tool and replace with Claude's descriptor
        tools = [t for t in request.tools if getattr(t, "name", None) != "bash"] + [
            _CLAUDE_BASH_DESCRIPTOR
        ]
        return handler(request.override(tools=tools))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async: replace our tool with Claude's bash descriptor."""
        # Filter out our registered bash tool and replace with Claude's descriptor
        tools = [t for t in request.tools if getattr(t, "name", None) != "bash"] + [
            _CLAUDE_BASH_DESCRIPTOR
        ]
        return await handler(request.override(tools=tools))

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
