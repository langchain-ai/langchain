"""Memory tool for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool)

This module provides a factory function for creating a memory tool that enables Claude
to store and retrieve information across conversations.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from langchain_anthropic.tools.types import MemoryCommand

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaMemoryTool20250818Param,
    )


class MemoryInput(BaseModel):
    """Input schema for memory tool."""

    command: Literal["view", "create", "str_replace", "insert", "delete", "rename"] = (
        Field(..., description="The command to execute")
    )

    path: str = Field(..., description="The path to the memory file")

    content: str | None = Field(
        default=None, description="The content for create command"
    )

    old_str: str | None = Field(
        default=None, description="The string to replace (for str_replace command)"
    )

    new_str: str | None = Field(
        default=None,
        description="The replacement string (for str_replace/insert commands)",
    )

    insert_line: int | None = Field(
        default=None, description="The line number to insert at (for insert command)"
    )

    new_path: str | None = Field(
        default=None, description="The new path (for rename command)"
    )


def memory_20250818(
    *,
    execute: Callable[[MemoryCommand], str | Awaitable[str]] | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaMemoryTool20250818Param | BaseTool:
    """Create a memory tool for persistent storage across conversations.

    The memory tool enables Claude to store and retrieve information across
    conversations through a memory file directory. Claude can create, read, update, and
    delete files that persist between sessions, allowing it to build knowledge over time
    without keeping everything in the context window.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool)
    for more details.

    Args:
        execute: Optional callback function for client-side execution.

            When provided, returns a `StructuredTool` that can be invoked locally. The
            function receives the command input and should return the result.

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
        model_with_memory = model.bind_tools(
            [tools.memory_20250818(cache_control={"type": "ephemeral"})],
        )

        response = model_with_memory.invoke("Remember that I like Python")
        ```

        Client-side execution (you execute the tool):

        ```python
        import os
        from langchain_anthropic import ChatAnthropic, tools

        MEMORY_DIR = "/tmp/memory"


        def execute_memory(args):
            path = os.path.join(MEMORY_DIR, args["path"])
            if args["command"] == "view":
                with open(path) as f:
                    return f.read()
            elif args["command"] == "create":
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(args["content"])
                return f"Created {args['path']}"
            elif args["command"] == "delete":
                os.remove(path)
                return f"Deleted {args['path']}"
            # Handle other commands...


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        memory_tool = tools.memory_20250818(execute=execute_memory)
        model_with_memory = model.bind_tools([memory_tool])
        ```
    """
    name = "memory"

    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "memory_20250818",
        "name": name,
    }
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

    # If no execute callback, return server-side definition
    if execute is None:
        return provider_tool_def  # type: ignore[return-value]

    # Create client-side tool with execute callback
    tool = StructuredTool.from_function(
        func=execute,
        name=name,
        description="Store and retrieve information across conversations. "
        "Commands: view, create, str_replace, insert, delete, rename.",
        args_schema=MemoryInput,
    )

    # Store provider-specific definition in extras
    tool.extras = {
        **(tool.extras or {}),
        "provider_tool_definition": provider_tool_def,
    }

    return tool


__all__ = ["memory_20250818"]
