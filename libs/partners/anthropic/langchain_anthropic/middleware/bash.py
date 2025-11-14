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

# Tool type constants
BASH_TOOL_TYPE = "bash_20250124"
BASH_TOOL_NAME = "bash"


class ClaudeBashToolMiddleware(ShellToolMiddleware):
    """Middleware that exposes Anthropic's native bash tool to models."""

    def __init__(
        self,
        workspace_root: str | None = None,
        *,
        tool_name: str = BASH_TOOL_NAME,
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
            tool_name: Name for the bash tool (default: "bash").
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
            shell_command=("/bin/bash",),
            env=env,
        )
        self.tool_name = tool_name

        # Create tool that will be executed by the tool node
        @tool(tool_name)
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
        tools = [
            t for t in request.tools if getattr(t, "name", None) != self.tool_name
        ] + [{"type": BASH_TOOL_TYPE, "name": self.tool_name}]
        return handler(request.override(tools=tools))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async: replace our tool with Claude's bash descriptor."""
        # Filter out our registered bash tool and replace with Claude's descriptor
        tools = [
            t for t in request.tools if getattr(t, "name", None) != self.tool_name
        ] + [{"type": BASH_TOOL_TYPE, "name": self.tool_name}]
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
            name=self.tool_name,
            status=status,
            artifact=artifact or {},
        )


__all__ = ["ClaudeBashToolMiddleware"]
