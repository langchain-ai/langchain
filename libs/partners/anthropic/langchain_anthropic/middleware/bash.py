"""Anthropic-specific middleware for the Claude bash tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.shell_tool import ShellToolMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
)

# Tool type constants for Anthropic
BASH_TOOL_TYPE = "bash_20250124"
BASH_TOOL_NAME = "bash"


class ClaudeBashToolMiddleware(ShellToolMiddleware):
    """Middleware that exposes Anthropic's native bash tool to models."""

    def __init__(
        self,
        workspace_root: str | None = None,
        *,
        startup_commands: tuple[str, ...] | list[str] | str | None = None,
        shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
        execution_policy: Any | None = None,
        redaction_rules: tuple[Any, ...] | list[Any] | None = None,
        tool_description: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> None:
        """Initialize middleware for Claude's native bash tool.

        Args:
            workspace_root: Base directory for the shell session.
                If omitted, a temporary directory is created.
            startup_commands: Optional commands executed after the session starts.
            shutdown_commands: Optional commands executed before session shutdown.
            execution_policy: Execution policy controlling timeouts and limits.
            redaction_rules: Optional redaction rules to sanitize output.
            tool_description: Optional override for tool description.
            env: Optional environment variables for the shell session.
        """
        super().__init__(
            workspace_root=workspace_root,
            startup_commands=startup_commands,
            shutdown_commands=shutdown_commands,
            execution_policy=execution_policy,
            redaction_rules=redaction_rules,
            tool_description=tool_description,
            tool_name=BASH_TOOL_NAME,
            shell_command=("/bin/bash",),
            env=env,
        )
        # Parent class now creates the tool with name "bash" via tool_name parameter

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Replace parent's shell tool with Claude's bash descriptor."""
        filtered = [
            t for t in request.tools if getattr(t, "name", None) != BASH_TOOL_NAME
        ]
        tools = [*filtered, {"type": BASH_TOOL_TYPE, "name": BASH_TOOL_NAME}]
        return handler(request.override(tools=tools))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async: replace parent's shell tool with Claude's bash descriptor."""
        filtered = [
            t for t in request.tools if getattr(t, "name", None) != BASH_TOOL_NAME
        ]
        tools = [*filtered, {"type": BASH_TOOL_TYPE, "name": BASH_TOOL_NAME}]
        return await handler(request.override(tools=tools))


__all__ = ["ClaudeBashToolMiddleware"]
