"""Bash tool for Claude models.

[Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)

This module provides a factory function for creating a bash tool that enables Claude to
execute shell commands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from langchain_anthropic.tools.types import BashCommand

if TYPE_CHECKING:
    from anthropic.types import CacheControlEphemeralParam, ToolBash20250124Param


class BashInput(BaseModel):
    """Input schema for bash tool."""

    command: str | None = Field(default=None, description="The bash command to run")

    restart: bool | None = Field(
        default=None, description="Set to true to restart the bash session"
    )


def bash_20250124(
    *,
    execute: Callable[[BashCommand], str | Awaitable[str]] | None = None,
    cache_control: CacheControlEphemeralParam | None = None,
) -> ToolBash20250124Param | BaseTool:
    """Create a bash tool for executing shell commands.

    The bash tool enables Claude to execute bash commands in a persistent
    shell session. Claude can run system operations, install packages,
    and interact with the filesystem.

    Supported models:

    - Claude 4 models
    - Sonnet 3.7 (deprecated)

    See the [Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)
    for more details.

    Note:
        This tool is typically used in conjunction with the computer use tool for
        agentic workflows.

    Args:
        execute: Optional callback function for client-side execution.

            When provided, returns a `StructuredTool` that can be invoked locally. The
            function receives the command input and should return the result (stdout and
            stderr combined, or an error message).

            If not provided, returns a server-side tool definition that Anthropic
            executes.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        If `execute` is provided: A `StructuredTool` that can be invoked locally
            and passed to `bind_tools`.

        If `execute` is not provided: A server-side tool definition dict to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        Server-side execution (Anthropic executes the tool):

        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_bash = model.bind_tools(
            [tools.bash_20250124(cache_control={"type": "ephemeral"})],
        )

        response = model_with_bash.invoke("List the files in the current directory")
        ```

        Client-side execution (you execute the tool):

        ```python
        import subprocess
        from langchain_anthropic import ChatAnthropic, tools


        def execute_bash(args):
            if args.get("restart"):
                return "Bash session restarted"
            try:
                result = subprocess.run(
                    args["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout + result.stderr
            except Exception as e:
                return f"Error: {e}"


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        bash_tool = tools.bash_20250124(execute=execute_bash)
        model_with_bash = model.bind_tools([bash_tool])

        response = model_with_bash.invoke("List Python files in the current directory")
        # Process tool calls and invoke bash_tool with the args
        ```
    """
    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "bash_20250124",
        "name": "bash",
    }
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

    # If no execute callback, return server-side definition
    if execute is None:
        return provider_tool_def  # type: ignore[return-value]

    # Create client-side tool with execute callback
    tool = StructuredTool.from_function(
        func=execute,
        name="bash",
        description="Execute bash commands in a persistent shell session. "
        "Use `command` to run a shell command, or `restart: true` to reset "
        "the session.",
        args_schema=BashInput,
    )

    # Store provider-specific definition in extras
    tool.extras = {
        **(tool.extras or {}),
        "provider_tool_definition": provider_tool_def,
    }

    return tool


__all__ = ["bash_20250124"]
