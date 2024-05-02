from typing import TYPE_CHECKING, Any

from langchain._api import create_importer
from langchain.retrievers.pubmed import PubMedRetriever

if TYPE_CHECKING:
    from langchain_community.retrievers import PubMedRetriever

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {"PubMedRetriever": "langchain_community.retrievers"}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "PubMedRetriever",
]
