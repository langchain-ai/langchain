"""Text editor tool for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)

This module provides factory functions for creating text editor tools that enable Claude
to view and modify files.

`text_editor_20241022` was for Claude Sonnet 3.5 and is now retired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types import (
        CacheControlEphemeralParam,
        ToolTextEditor20250124Param,
        ToolTextEditor20250429Param,
        ToolTextEditor20250728Param,
    )


def text_editor_20250728(
    *,
    cache_control: CacheControlEphemeralParam | None = None,
) -> ToolTextEditor20250728Param:
    """Create a text editor tool for Claude 4.x models.

    The text editor tool enables Claude to view and modify files using string
    replacement operations.

    This is the latest version of the text editor tool.

    Supported models:

    - Claude 4.x models.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)
    for more details.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A text editor tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_editor = model.bind_tools(
            [tools.text_editor_20250728(cache_control={"type": "ephemeral"})],
        )

        response = model_with_editor.invoke("View the contents of main.py")
        ```
    """
    if cache_control is not None:
        return {
            "type": "text_editor_20250728",
            "name": "str_replace_based_edit_tool",
            "cache_control": cache_control,
        }
    return {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
    }


def text_editor_20250429(
    *,
    cache_control: CacheControlEphemeralParam | None = None,
) -> ToolTextEditor20250429Param:
    """Create a text editor tool (April 2025 version).

    Older version of `text_editor_20250728`. You should prefer using that
    function unless you need to use this specific version.

    The text editor tool enables Claude to view and modify files using string
    replacement operations.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)
    for more details.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A text editor tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].
    """
    if cache_control is not None:
        return {
            "type": "text_editor_20250429",
            "name": "str_replace_based_edit_tool",
            "cache_control": cache_control,
        }
    return {
        "type": "text_editor_20250429",
        "name": "str_replace_based_edit_tool",
    }


def text_editor_20250124(
    *,
    cache_control: CacheControlEphemeralParam | None = None,
) -> ToolTextEditor20250124Param:
    """Create a text editor tool (January 2025 version).

    The text editor tool enables Claude to view and modify files using string
    replacement operations.

    Supported models:

    - Claude Sonnet 3.7 (deprecated)

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool)
    for more details.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A text editor tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].
    """
    if cache_control is not None:
        return {
            "type": "text_editor_20250124",
            "name": "str_replace_editor",
            "cache_control": cache_control,
        }
    return {
        "type": "text_editor_20250124",
        "name": "str_replace_editor",
    }


__all__ = [
    "text_editor_20250124",
    "text_editor_20250429",
    "text_editor_20250728",
]
