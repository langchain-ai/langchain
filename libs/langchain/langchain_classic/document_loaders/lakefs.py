from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import LakeFSLoader
    from langchain_community.document_loaders.lakefs import (
        LakeFSClient,
        UnstructuredLakeFSLoader,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "LakeFSClient": "langchain_community.document_loaders.lakefs",
    "LakeFSLoader": "langchain_community.document_loaders",
    "UnstructuredLakeFSLoader": "langchain_community.document_loaders.lakefs",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "LakeFSClient",
    "LakeFSLoader",
    "UnstructuredLakeFSLoader",
]
