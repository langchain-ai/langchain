"""Web search tool for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool)

This module provides a factory function for creating a web search tool that gives Claude
access to real-time web content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types import CacheControlEphemeralParam, WebSearchTool20250305Param
    from anthropic.types.web_search_tool_20250305_param import UserLocation


def web_search_20250305(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: UserLocation | None = None,
    cache_control: CacheControlEphemeralParam | None = None,
) -> WebSearchTool20250305Param:
    """Create a web search tool that gives Claude access to real-time web content.

    The web search tool allows Claude to perform real-time web searches
    to find current information beyond its training data cutoff.

    Claude automatically cites sources from search results.

    See the [Claude docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool)
    for more details.

    Args:
        max_uses: Maximum number of times the tool can be used in the API request.
        allowed_domains: Only include results from these domains.

            Cannot be used with `blocked_domains`.
        blocked_domains: Never include results from these domains.

            Cannot be used with `allowed_domains`.
        user_location: User's approximate location for more relevant results.

            Should be a dict with `type` set to `'approximate'` and optional
            keys: `city`, `region`, `country` (ISO 3166-1 alpha-2), `timezone`.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A web search tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_search = model.bind_tools(
            [tools.web_search_20250305(cache_control={"type": "ephemeral"})],
        )

        response = model_with_search.invoke("What is the weather in NYC?")
        ```

        ```python title="With options"
        model_with_search = model.bind_tools(
            [
                tools.web_search_20250305(
                    max_uses=5,
                    allowed_domains=["reuters.com", "bbc.com"],
                    user_location={
                        "type": "approximate",
                        "city": "San Francisco",
                        "region": "California",
                        "country": "US",
                        "timezone": "America/Los_Angeles",
                    },
                )
            ]
        )
        ```
    """
    result: WebSearchTool20250305Param = {
        "type": "web_search_20250305",
        "name": "web_search",
    }
    if max_uses is not None:
        result["max_uses"] = max_uses
    if allowed_domains is not None:
        result["allowed_domains"] = allowed_domains
    if blocked_domains is not None:
        result["blocked_domains"] = blocked_domains
    if user_location is not None:
        result["user_location"] = user_location
    if cache_control is not None:
        result["cache_control"] = cache_control
    return result


__all__ = ["web_search_20250305"]
