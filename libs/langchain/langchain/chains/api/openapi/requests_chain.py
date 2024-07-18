from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.openapi.requests_chain import (
        REQUEST_TEMPLATE,
        APIRequesterChain,
        APIRequesterOutputParser,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "APIRequesterChain": "langchain_community.chains.openapi.requests_chain",
    "APIRequesterOutputParser": "langchain_community.chains.openapi.requests_chain",
    "REQUEST_TEMPLATE": "langchain_community.chains.openapi.requests_chain",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = ["APIRequesterChain", "APIRequesterOutputParser", "REQUEST_TEMPLATE"]
