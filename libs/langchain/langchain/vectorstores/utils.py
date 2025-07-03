from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores.utils import (
        DistanceStrategy, filter_complex_metadata, maximal_marginal_relevance)

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "DistanceStrategy": "langchain_community.vectorstores.utils",
    "maximal_marginal_relevance": "langchain_community.vectorstores.utils",
    "filter_complex_metadata": "langchain_community.vectorstores.utils",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "DistanceStrategy",
    "maximal_marginal_relevance",
    "filter_complex_metadata",
]
