"""Memory tool for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool)

This module provides a factory function for creating a memory tool that enables Claude
to store and retrieve information across conversations.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from anthropic.types.beta import BetaCacheControlEphemeralParam


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
    execute: Callable[..., str | Awaitable[str]],
    *,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> StructuredTool:
    r"""Create a memory tool for persistent storage across conversations.

    The memory tool enables Claude to store and retrieve information across
    conversations through a memory file directory. Claude can create, read, update, and
    delete files that persist between sessions, allowing it to build knowledge over time
    without keeping everything in the context window.

    !!! warning "Security advisory"

        This tool can create, read, modify, and delete files on your system. Do not
        use with untrusted input. The `execute` callback you provide should restrict
        file operations to a dedicated memory directory and validate all paths.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool)
    for more details.

    Args:
        execute: Callback function for executing memory commands.

            See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool#tool-commands)
            for the available commands.

            Can be sync or async.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A `StructuredTool` that can be invoked locally and passed to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python title="Manual tool execution loop"
        import os
        import tempfile

        from langchain_anthropic import ChatAnthropic, tools
        from langchain_core.messages import HumanMessage, ToolMessage

        # Create a temporary workspace directory for this demo.
        # In production, use a persistent directory path.
        memory_dir = tempfile.mkdtemp(prefix="memory-")


        def execute_memory(
            *,
            command: str,
            path: str,
            content: str | None = None,
            old_str: str | None = None,
            new_str: str | None = None,
            insert_line: int | None = None,
            new_path: str | None = None,
            **kw,
        ):
            # Claude sends absolute paths like "/memories/file.txt"
            # Strip leading "/" to make path relative for os.path.join
            full_path = os.path.join(memory_dir, path.lstrip("/"))
            if command == "view":
                if os.path.isdir(full_path):
                    files = os.listdir(full_path)
                    return f"Directory: {path}\\n" + "\\n".join(f"- {f}" for f in files)
                if not os.path.exists(full_path):
                    return f"Error: {path} does not exist"
                with open(full_path) as f:
                    content = f.read()
                lines = content.split("\\n")
                return "\\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
            elif command == "create":
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content or "")
                return f"Created {path}"
            elif command == "str_replace" and old_str and new_str:
                with open(full_path) as f:
                    file_content = f.read()
                file_content = file_content.replace(old_str, new_str)
                with open(full_path, "w") as f:
                    f.write(file_content)
                return f"Replaced text in {path}"
            elif command == "insert" and insert_line and new_str:
                with open(full_path) as f:
                    lines = f.readlines()
                lines.insert(insert_line - 1, new_str + "\\n")
                with open(full_path, "w") as f:
                    f.writelines(lines)
                return f"Inserted at line {insert_line} in {path}"
            elif command == "delete":
                os.remove(full_path)
                return f"Deleted {path}"
            elif command == "rename" and new_path is not None:
                new_full_path = os.path.join(memory_dir, new_path.lstrip("/"))
                os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
                os.rename(full_path, new_full_path)
                return f"Renamed {path} to {new_path}"
            return "Command executed"


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        memory_tool = tools.memory_20250818(execute=execute_memory)
        model_with_memory = model.bind_tools([memory_tool])

        query = HumanMessage(content="Remember that my favorite color is blue")
        response = model_with_memory.invoke([query])

        # Process tool calls in a loop until no more tool calls
        messages = [query, response]

        while response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"Memory command: {tool_call['args'].get('command')}")
                result = memory_tool.invoke(tool_call["args"])
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
                messages.append(tool_msg)

            response = model_with_memory.invoke(messages)
            messages.append(response)

        print(response.content)
        ```

        ```python title="Automatic tool execution"
        import os
        import tempfile

        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic, tools

        # Create a temporary workspace directory for this demo.
        # In production, use a persistent directory path.
        memory_dir = tempfile.mkdtemp(prefix="memory-")


        def execute_memory(
            *,
            command: str,
            path: str,
            content: str | None = None,
            old_str: str | None = None,
            new_str: str | None = None,
            insert_line: int | None = None,
            new_path: str | None = None,
            **kw,
        ):
            # Claude sends absolute paths like "/memories/file.txt"
            # Strip leading "/" to make path relative for os.path.join
            full_path = os.path.join(memory_dir, path.lstrip("/"))
            if command == "view":
                if os.path.isdir(full_path):
                    files = os.listdir(full_path)
                    return f"Directory: {path}\\n" + "\\n".join(f"- {f}" for f in files)
                if not os.path.exists(full_path):
                    return f"Error: {path} does not exist"
                with open(full_path) as f:
                    return f.read()
            elif command == "create":
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content or "")
                return f"Created {path}"
            elif command == "str_replace" and old_str is not None:
                with open(full_path) as f:
                    file_content = f.read()
                file_content = file_content.replace(old_str, new_str or "")
                with open(full_path, "w") as f:
                    f.write(file_content)
                return f"Replaced text in {path}"
            elif command == "delete":
                os.remove(full_path)
                return f"Deleted {path}"
            return "Command executed"


        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
            tools=[tools.memory_20250818(execute=execute_memory)],
        )

        query = {"messages": [{"role": "user", "content": "Remember my name is Alice"}]}
        result = agent.invoke(query)

        for message in result["messages"]:
            message.pretty_print()
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
