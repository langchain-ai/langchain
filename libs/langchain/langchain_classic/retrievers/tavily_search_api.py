from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.retrievers import TavilySearchAPIRetriever
    from langchain_community.retrievers.tavily_search_api import SearchDepth

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "SearchDepth": "langchain_community.retrievers.tavily_search_api",
    "TavilySearchAPIRetriever": "langchain_community.retrievers",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "SearchDepth",
    "TavilySearchAPIRetriever",
]
