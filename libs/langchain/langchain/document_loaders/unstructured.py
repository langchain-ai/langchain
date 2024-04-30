from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        UnstructuredAPIFileIOLoader,
        UnstructuredAPIFileLoader,
        UnstructuredFileIOLoader,
        UnstructuredFileLoader,
    )
    from langchain_community.document_loaders.unstructured import (
        UnstructuredBaseLoader,
        get_elements_from_api,
        satisfies_min_unstructured_version,
        validate_unstructured_version,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "satisfies_min_unstructured_version": (
        "langchain_community.document_loaders.unstructured"
    ),
    "validate_unstructured_version": (
        "langchain_community.document_loaders.unstructured"
    ),
    "UnstructuredBaseLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredFileLoader": "langchain_community.document_loaders",
    "get_elements_from_api": "langchain_community.document_loaders.unstructured",
    "UnstructuredAPIFileLoader": "langchain_community.document_loaders",
    "UnstructuredFileIOLoader": "langchain_community.document_loaders",
    "UnstructuredAPIFileIOLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "satisfies_min_unstructured_version",
    "validate_unstructured_version",
    "UnstructuredBaseLoader",
    "UnstructuredFileLoader",
    "get_elements_from_api",
    "UnstructuredAPIFileLoader",
    "UnstructuredFileIOLoader",
    "UnstructuredAPIFileIOLoader",
]
