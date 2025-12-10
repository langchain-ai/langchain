"""Memory tool for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool)

This module provides a factory function for creating a memory tool that enables Claude
to store and retrieve information across conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaMemoryTool20250818Param,
    )


def memory_20250818(
    *,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaMemoryTool20250818Param:
    """Create a memory tool for persistent storage across conversations.

    The memory tool enables Claude to store and retrieve information across
    conversations through a memory file directory. Claude can create, read, update, and
    delete files that persist between sessions, allowing it to build knowledge over time
    without keeping everything in the context window.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool)
    for more details.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A memory tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_memory = model.bind_tools(
            [tools.memory_20250818(cache_control={"type": "ephemeral"})],
        )

        response = model_with_memory.invoke("Remember that I like Python")
        ```
    """
    if cache_control is not None:
        return {
            "type": "memory_20250818",
            "name": "memory",
            "cache_control": cache_control,
        }
    return {
        "type": "memory_20250818",
        "name": "memory",
    }


__all__ = ["memory_20250818"]
