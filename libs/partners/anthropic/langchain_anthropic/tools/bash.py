"""Bash tool for Claude models.

[Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)

This module provides a factory function for creating a bash tool that enables Claude to
execute shell commands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from anthropic.types import CacheControlEphemeralParam


class BashInput(BaseModel):
    """Input schema for bash tool."""

    command: str | None = Field(default=None, description="The bash command to run")

    restart: bool | None = Field(
        default=None, description="Set to true to restart the bash session"
    )


def bash_20250124(
    execute: Callable[..., str | Awaitable[str]],
    *,
    cache_control: CacheControlEphemeralParam | None = None,
) -> StructuredTool:
    """Create a bash tool for executing shell commands.

    The bash tool enables Claude to execute bash commands in a persistent
    shell session. Claude can run system operations, install packages,
    and interact with the filesystem.

    Supported models:

    - Claude 4 models
    - Sonnet 3.7 (deprecated)

    !!! warning "Security advisory"

        This tool can execute arbitrary shell commands on your system. Do not
        use with untrusted input. Run in a sandboxed environment (container or VM)
        with minimal privileges. The `execute` callback you provide is responsible
        for any safety measures such as command filtering or resource limits.

    See the [Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)
    for more details.

    Note:
        This tool is typically used in conjunction with the computer use tool for
        agentic workflows.

    Args:
        execute: Callback function for executing bash commands.

            See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/bash-tool#implement-the-bash-tool)
            for implementation details.

            Can be sync or async.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A `StructuredTool` that can be invoked locally and passed to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python title="Manual tool execution loop"
        import subprocess

        from langchain_anthropic import ChatAnthropic, tools
        from langchain.messages import HumanMessage, ToolMessage


        def execute_bash(*, command: str | None = None, restart: bool = False, **kw):
            if restart:
                return "Bash session restarted"
            try:
                result = subprocess.run(
                    command,
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

        query = HumanMessage(content="List files in the current directory")
        response = model_with_bash.invoke([query])

        # Process tool calls in a loop until no more tool calls
        messages = [query, response]

        while response.tool_calls:
            for tool_call in response.tool_calls:
                result = bash_tool.invoke(tool_call["args"])
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
                messages.append(tool_msg)

            response = model_with_bash.invoke(messages)
            messages.append(response)

        print(response.content)
        ```

        ```python title="Automatic tool execution"
        import subprocess

        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic, tools


        def execute_bash(*, command: str | None = None, restart: bool = False, **kw):
            if restart:
                return "Bash session restarted"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout + result.stderr


        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
            tools=[tools.bash_20250124(execute=execute_bash)],
        )

        result = agent.invoke({"messages": [{"role": "user", "content": "List files"}]})
        ```
    """
    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "bash_20250124",
        "name": "bash",
    }
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

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
