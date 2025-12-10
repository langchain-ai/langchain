"""Tool search tools for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/tool-search-tool)

This module provides factory functions for creating tool search tools that enable Claude
to dynamically discover and load tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaToolSearchToolBm25_20251119Param,
        BetaToolSearchToolRegex20251119Param,
    )


def tool_search_regex_20251119(
    *,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaToolSearchToolRegex20251119Param:
    """Create a regex-based tool search tool.

    The tool search tool enables Claude to dynamically discover and load tools on
    demand. This version uses regex matching for tool discovery.

    Use this with `defer_loading=True` on other tools to create large tool sets where
    tools are only loaded when needed.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/tool-search-tool)
    for more details.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A tool search tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_search = model.bind_tools(
            [
                tools.tool_search_regex_20251119(
                    cache_control={"type": "ephemeral"},
                ),
                # Other tools with defer_loading=True in extras
            ]
        )
        ```
    """
    if cache_control is not None:
        return {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex",
            "cache_control": cache_control,
        }
    return {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }


def tool_search_bm25_20251119(
    *,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaToolSearchToolBm25_20251119Param:
    """Create a BM25-based tool search tool.

    The tool search tool enables Claude to dynamically discover and load tools on
    demand. This version uses BM25 ranking for tool discovery, which is generally more
    accurate for natural language queries.

    Use this with `defer_loading=True` on other tools to create large tool sets where
    tools are only loaded when needed.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/tool-search-tool)
    for more details.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A tool search tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_search = model.bind_tools(
            [
                tools.tool_search_bm25_20251119(
                    cache_control={"type": "ephemeral"},
                ),
                # Other tools with defer_loading=True in extras
            ]
        )
        ```
    """
    if cache_control is not None:
        return {
            "type": "tool_search_tool_bm25_20251119",
            "name": "tool_search_tool_bm25",
            "cache_control": cache_control,
        }
    return {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    }


__all__ = ["tool_search_bm25_20251119", "tool_search_regex_20251119"]
