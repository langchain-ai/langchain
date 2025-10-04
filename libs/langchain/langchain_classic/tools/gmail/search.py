from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import GmailSearch
    from langchain_community.tools.gmail.search import Resource, SearchArgsSchema

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "Resource": "langchain_community.tools.gmail.search",
    "SearchArgsSchema": "langchain_community.tools.gmail.search",
    "GmailSearch": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "GmailSearch",
    "Resource",
    "SearchArgsSchema",
]
