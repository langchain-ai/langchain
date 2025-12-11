"""Text editor tool for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)

This module provides factory functions for creating text editor tools that enable Claude
to view and modify files.

`text_editor_20241022` was for Claude Sonnet 3.5 and is now retired.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from anthropic.types import CacheControlEphemeralParam


class TextEditorInput(BaseModel):
    """Input schema for text editor tool."""

    command: Literal["view", "str_replace", "create", "insert"] = Field(
        ..., description="The command to execute"
    )

    path: str = Field(..., description="The path to the file or directory")

    view_range: tuple[int, int] | None = Field(
        default=None, description="Optional (start_line, end_line) range for view"
    )

    old_str: str | None = Field(
        default=None, description="The string to replace (for str_replace command)"
    )

    new_str: str | None = Field(
        default=None,
        description="The replacement string (for str_replace/insert commands)",
    )

    file_text: str | None = Field(
        default=None, description="The file content (for create command)"
    )

    insert_line: int | None = Field(
        default=None, description="The line number to insert at (for insert command)"
    )


def text_editor_20250728(
    execute: Callable[..., str | Awaitable[str]],
    *,
    max_characters: int | None = None,
    cache_control: CacheControlEphemeralParam | None = None,
) -> StructuredTool:
    r"""Create a text editor tool for Claude 4.x models.

    The text editor tool enables Claude to view and modify files using string
    replacement operations.

    This is the latest version of the text editor tool.

    Supported models:

    - Claude 4.x models.

    !!! warning

        This tool can read and write files on your system. Do not use with
        untrusted input. The `execute` callback you provide should validate file
        paths and restrict access to intended directories only.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)
    for more details.

    Args:
        execute: Callback function for executing text editor commands.

            See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool#implement-the-text-editor-tool)
            for the available commands.

            Can be sync or async.
        max_characters: Optional maximum characters to return when viewing files.

            If the file content exceeds this limit, it will be truncated.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A `StructuredTool` that can be invoked locally and passed to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python title="Manual tool execution loop"
        from langchain_anthropic import ChatAnthropic, tools
        from langchain.messages import HumanMessage, ToolMessage


        def execute_editor(*, command, path, old_str=None, new_str=None, **kw):
            if command == "view":
                with open(path) as f:
                    content = f.read()
                lines = content.split("\\n")
                return "\\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
            elif command == "str_replace":
                with open(path) as f:
                    content = f.read()
                content = content.replace(old_str, new_str)
                with open(path, "w") as f:
                    f.write(content)
                return "Successfully replaced text."
            # Handle other commands...
            return "Command executed"


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        editor_tool = tools.text_editor_20250728(execute=execute_editor)
        model_with_editor = model.bind_tools([editor_tool])

        query = HumanMessage(content="View the contents of ./config.py")
        response = model_with_editor.invoke([query])

        # Process tool calls in a loop until no more tool calls
        messages = [query, response]

        while response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"Tool call args: {tool_call['args']}")
                result = editor_tool.invoke(tool_call["args"])
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
                messages.append(tool_msg)

            response = model_with_editor.invoke(messages)
            messages.append(response)

        print(response.content)
        ```

        ```python title="Automatic tool execution"
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic, tools


        def execute_editor(*, command, path, old_str=None, new_str=None, **kw):
            if command == "view":
                with open(path) as f:
                    return f.read()
            elif command == "str_replace":
                with open(path) as f:
                    content = f.read()
                content = content.replace(old_str, new_str)
                with open(path, "w") as f:
                    f.write(content)
                return "Successfully replaced text."
            return "Command executed"


        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
            tools=[tools.text_editor_20250728(execute=execute_editor)],
        )

        query = {"messages": [{"role": "user", "content": "View ./config.py"}]}
        result = agent.invoke(query)

        for message in result["messages"]:
            message.pretty_print()
        ```
    """
    name = "str_replace_based_edit_tool"

    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "text_editor_20250728",
        "name": name,
    }
    if max_characters is not None:
        provider_tool_def["max_characters"] = max_characters
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

    # Create client-side tool with execute callback
    tool = StructuredTool.from_function(
        func=execute,
        name=name,
        description="View and modify text files. Commands: view (examine file), "
        "str_replace (replace text), create (new file), insert (add text at line).",
        args_schema=TextEditorInput,
    )

    # Store provider-specific definition in extras
    tool.extras = {
        **(tool.extras or {}),
        "provider_tool_definition": provider_tool_def,
    }

    return tool


def text_editor_20250429(
    execute: Callable[..., str | Awaitable[str]],
    *,
    cache_control: CacheControlEphemeralParam | None = None,
) -> StructuredTool:
    """Create a text editor tool (April 2025 version).

    Older version of `text_editor_20250728`. You should prefer using that
    function unless you need to use this specific version.

    The text editor tool enables Claude to view and modify files using string
    replacement operations.

    !!! warning

        This tool can read and write files on your system. Do not use with
        untrusted input. The `execute` callback you provide should validate file
        paths and restrict access to intended directories only.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)
    for more details.

    Args:
        execute: Callback function for executing text editor commands.

            See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool#implement-the-text-editor-tool)
            for the available commands.

            Can be sync or async.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A `StructuredTool` that can be invoked locally and passed to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].
    """
    name = "str_replace_based_edit_tool"

    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "text_editor_20250429",
        "name": name,
    }
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

    # Create client-side tool with execute callback
    tool = StructuredTool.from_function(
        func=execute,
        name=name,
        description="View and modify text files. Commands: view (examine file), "
        "str_replace (replace text), create (new file), insert (add text at line).",
        args_schema=TextEditorInput,
    )

    # Store provider-specific definition in extras
    tool.extras = {
        **(tool.extras or {}),
        "provider_tool_definition": provider_tool_def,
    }

    return tool


def text_editor_20250124(
    execute: Callable[..., str | Awaitable[str]],
    *,
    cache_control: CacheControlEphemeralParam | None = None,
) -> StructuredTool:
    """Create a text editor tool (January 2025 version).

    The text editor tool enables Claude to view and modify files using string
    replacement operations.

    Supported models:

    - Claude Sonnet 3.7 (deprecated)

    !!! warning

        This tool can read and write files on your system. Do not use with
        untrusted input. The `execute` callback you provide should validate file
        paths and restrict access to intended directories only.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)
    for more details.

    Args:
        execute: Callback function for executing text editor commands.

            See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool#implement-the-text-editor-tool)
            for the available commands.

            Can be sync or async.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A `StructuredTool` that can be invoked locally and passed to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].
    """
    name = "str_replace_editor"

    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "text_editor_20250124",
        "name": name,
    }
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

    # Create client-side tool with execute callback
    tool = StructuredTool.from_function(
        func=execute,
        name=name,
        description="View and modify text files. Commands: view (examine file), "
        "str_replace (replace text), create (new file), insert (add text at line).",
        args_schema=TextEditorInput,
    )

    # Store provider-specific definition in extras
    tool.extras = {
        **(tool.extras or {}),
        "provider_tool_definition": provider_tool_def,
    }

    return tool


__all__ = [
    "text_editor_20250124",
    "text_editor_20250429",
    "text_editor_20250728",
]
