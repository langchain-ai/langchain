"""Web fetch tool for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-fetch-tool)

This module provides a factory function for creating a web fetch tool
that allows Claude to retrieve full content from specified web pages and PDFs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaCitationsConfigParam,
        BetaWebFetchTool20250910Param,
    )


class CitationsConfig(TypedDict, total=False):
    """Configuration for citations in web fetch results.

    Attributes:
        enabled: Whether to enable citations for fetched content.
    """

    enabled: bool


def web_fetch_20250910(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    citations: CitationsConfig | None = None,
    max_content_tokens: int | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaWebFetchTool20250910Param:
    """Create a web fetch tool that retrieves content from web pages and PDFs.

    The web fetch tool allows Claude to retrieve full content from specified web pages
    and PDF documents.

    Claude can only fetch URLs that have been explicitly provided by the user or that
    come from previous web search or web fetch results.

    !!! warning

        Enabling the web fetch tool in environments where Claude processes
        untrusted input alongside sensitive data poses data exfiltration risks.
        Only use this tool in trusted environments or when handling non-sensitive data.

    See the [Claude docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-fetch-tool)
    for more details.

    Args:
        max_uses: Maximum number of times the tool can be used in the API request.
        allowed_domains: Only fetch from these domains.

            Cannot be used with `blocked_domains`.
        blocked_domains: Never fetch from these domains.

            Cannot be used with `allowed_domains`.
        citations: Enable citations for fetched content.

            Use `{'enabled': True}` to enable. Unlike web search where citations
            are always enabled, citations are optional for web fetch.
        max_content_tokens: Maximum content length in tokens.

            If the fetched content exceeds this limit, it will be truncated. This helps
            control token usage when fetching large documents.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A web fetch tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_fetch = model.bind_tools(
            [tools.web_fetch_20250910(cache_control={"type": "ephemeral"})],
        )

        response = model_with_fetch.invoke("Analyze the content at https://example.com/article")
        ```

        ```python title="With options"
        model_with_fetch = model.bind_tools(
            [
                tools.web_fetch_20250910(
                    max_uses=5,
                    allowed_domains=["arxiv.org", "example.com"],
                    citations={"enabled": True},
                    max_content_tokens=50000,
                )
            ]
        )
        ```

        ```python title="Combined with web search"
        model_with_tools = model.bind_tools(
            [
                tools.web_search_20250305(max_uses=3),
                tools.web_fetch_20250910(
                    max_uses=5,
                    citations={"enabled": True},
                ),
            ]
        )

        response = model_with_tools.invoke("Find quantum computing articles")
        ```
    """
    result: BetaWebFetchTool20250910Param = {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
    }
    if max_uses is not None:
        result["max_uses"] = max_uses
    if allowed_domains is not None:
        result["allowed_domains"] = allowed_domains
    if blocked_domains is not None:
        result["blocked_domains"] = blocked_domains
    if citations is not None:
        _citations: BetaCitationsConfigParam = {}
        if "enabled" in citations:
            _citations["enabled"] = citations["enabled"]
        result["citations"] = _citations
    if max_content_tokens is not None:
        result["max_content_tokens"] = max_content_tokens
    if cache_control is not None:
        result["cache_control"] = cache_control
    return result


__all__ = ["CitationsConfig", "web_fetch_20250910"]
