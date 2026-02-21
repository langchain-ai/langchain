"""Utilities for converting standard builtin tools to xAI format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from langchain_core.tools.builtin import (
        CodeExecutionTool,
        WebSearchTool,
        XSearchTool,
    )


def convert_standard_to_xai(
    tool: WebSearchTool | CodeExecutionTool | XSearchTool | dict[str, Any],
) -> dict[str, Any] | None:
    """Convert standard builtin tool to xAI format.

    Args:
        tool: Standard builtin tool definition with 'type' field.

    Returns:
        xAI-formatted tool definition, or None if tool type is not supported
        by xAI.

    Examples:
        >>> convert_standard_to_xai({"type": "web_search"})
        {'type': 'web_search'}

        >>> convert_standard_to_xai({"type": "code_execution"})
        {'type': 'code_interpreter'}

        >>> convert_standard_to_xai({"type": "x_search"})
        {'type': 'x_search'}
    """
    # Cast to dict for easier key access across union types
    tool_dict = cast("dict[str, Any]", tool)
    tool_type = tool_dict.get("type")

    if tool_type == "web_search":
        # xAI web search format
        return {"type": "web_search"}

    if tool_type == "code_execution":
        # xAI uses "code_interpreter" instead of "code_execution"
        return {"type": "code_interpreter"}

    if tool_type == "x_search":
        # xAI X/Twitter search format (unique to xAI)
        return {"type": "x_search"}

    # Tool not supported by xAI
    # (web_fetch, memory, file_search, image_generation, text_editor,
    # bash are not supported)
    return None
