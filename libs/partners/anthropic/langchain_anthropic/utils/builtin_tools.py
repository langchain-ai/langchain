"""Utilities for converting standard builtin tools to Anthropic format."""

from __future__ import annotations

from typing import Any, Union

from langchain_core.tools.builtin import (
    BashTool,
    CodeExecutionTool,
    MemoryTool,
    TextEditorTool,
    WebFetchTool,
    WebSearchTool,
)


def convert_standard_to_anthropic(
    tool: Union[
        WebSearchTool,
        CodeExecutionTool,
        WebFetchTool,
        MemoryTool,
        TextEditorTool,
        BashTool,
        dict[str, Any],
    ],
) -> dict[str, Any] | None:
    """Convert standard builtin tool to Anthropic format.

    Args:
        tool: Standard builtin tool definition with 'type' field.

    Returns:
        Anthropic-formatted tool definition, or None if tool type is not supported
        by Anthropic.

    Examples:
        >>> convert_standard_to_anthropic({"type": "web_search"})
        {'type': 'web_search_20250305', 'name': 'web_search'}

        >>> convert_standard_to_anthropic({"type": "code_execution"})
        {'type': 'code_execution_20250825', 'name': 'code_execution'}

        >>> convert_standard_to_anthropic({
        ...     "type": "web_search",
        ...     "max_uses": 5
        ... })
        {'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': 5}
    """
    tool_type = tool.get("type")

    if tool_type == "web_search":
        # Anthropic web search format
        result: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",
        }
        # Pass through max_uses if provided
        if "max_uses" in tool:
            result["max_uses"] = tool["max_uses"]
        # Pass through user_location if provided
        if "user_location" in tool:
            result["user_location"] = tool["user_location"]
        return result

    if tool_type == "code_execution":
        # Anthropic code execution format
        return {
            "type": "code_execution_20250825",
            "name": "code_execution",
        }

    if tool_type == "web_fetch":
        # Anthropic web fetch format
        result = {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
        }
        # Pass through max_uses if provided
        if "max_uses" in tool:
            result["max_uses"] = tool["max_uses"]
        return result

    if tool_type == "memory":
        # Anthropic memory format
        return {
            "type": "memory_20250818",
            "name": "memory",
        }

    if tool_type == "text_editor":
        # Anthropic text editor format
        return {
            "type": "text_editor_20250728",
            "name": "str_replace_based_edit_tool",
        }

    if tool_type == "bash":
        # Anthropic bash format
        return {
            "type": "bash_20250124",
            "name": "bash",
        }

    # Tool not supported by Anthropic
    # (file_search, image_generation, x_search are not supported)
    return None

