"""Tavily Search API toolkit."""

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.tavily_search.tool import (
        TavilyAnswer,
        TavilySearchResults,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "TavilySearchResults": "langchain_community.tools.tavily_search.tool",
    "TavilyAnswer": "langchain_community.tools.tavily_search.tool",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "TavilyAnswer",
    "TavilySearchResults",
]
