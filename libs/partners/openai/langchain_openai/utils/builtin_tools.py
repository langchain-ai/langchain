"""Utilities for converting standard builtin tools to OpenAI format."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.tools.builtin import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    WebSearchTool,
    XSearchTool,
)


def convert_standard_to_openai(
    tool: WebSearchTool
    | CodeExecutionTool
    | FileSearchTool
    | ImageGenerationTool
    | XSearchTool
    | dict[str, Any],
) -> dict[str, Any] | None:
    """Convert standard builtin tool to OpenAI format.

    Args:
        tool: Standard builtin tool definition with 'type' field.

    Returns:
        OpenAI-formatted tool definition, or None if tool type is not supported
        by OpenAI.

    Examples:
        >>> convert_standard_to_openai({"type": "web_search"})
        {'type': 'web_search'}

        >>> convert_standard_to_openai({"type": "code_execution"})
        {'type': 'code_interpreter', 'container': {'type': 'auto'}}

        >>> convert_standard_to_openai(
        ...     {
        ...         "type": "web_search",
        ...         "user_location": {"type": "approximate", "city": "San Francisco"},
        ...     }
        ... )
        {'type': 'web_search', 'user_location': {'type': 'approximate',
        'city': 'San Francisco'}}
    """
    # Cast to dict for easier key access across union types
    tool_dict = cast(dict[str, Any], tool)
    tool_type = tool_dict.get("type")

    if tool_type == "web_search":
        # OpenAI web search format
        result: dict[str, Any] = {"type": "web_search"}
        # Pass through user_location if provided
        if "user_location" in tool_dict:
            result["user_location"] = tool_dict["user_location"]
        return result

    if tool_type == "code_execution":
        # OpenAI uses "code_interpreter" instead of "code_execution"
        result = {"type": "code_interpreter"}
        # Use provided container config or default to auto
        if "container" in tool_dict:
            result["container"] = tool_dict["container"]
        else:
            result["container"] = {"type": "auto"}
        return result

    if tool_type == "file_search":
        # OpenAI file search format
        result = {"type": "file_search"}
        # Pass through vector_store_ids if provided
        if "vector_store_ids" in tool_dict:
            result["vector_store_ids"] = tool_dict["vector_store_ids"]
        return result

    if tool_type == "image_generation":
        # OpenAI image generation format
        return {"type": "image_generation"}

    # Tool not supported by OpenAI
    # (web_fetch, memory, text_editor, bash are Anthropic-only)
    return None
