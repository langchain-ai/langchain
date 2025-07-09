"""Browser tools and toolkit."""

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        ClickTool,
        CurrentWebPageTool,
        ExtractHyperlinksTool,
        ExtractTextTool,
        GetElementsTool,
        NavigateBackTool,
        NavigateTool,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "NavigateTool": "langchain_community.tools",
    "NavigateBackTool": "langchain_community.tools",
    "ExtractTextTool": "langchain_community.tools",
    "ExtractHyperlinksTool": "langchain_community.tools",
    "GetElementsTool": "langchain_community.tools",
    "ClickTool": "langchain_community.tools",
    "CurrentWebPageTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "ClickTool",
    "CurrentWebPageTool",
    "ExtractHyperlinksTool",
    "ExtractTextTool",
    "GetElementsTool",
    "NavigateBackTool",
    "NavigateTool",
]
