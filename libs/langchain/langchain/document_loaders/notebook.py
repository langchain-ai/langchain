from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import NotebookLoader
    from langchain_community.document_loaders.notebook import (
        concatenate_cells,
        remove_newlines,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "concatenate_cells": "langchain_community.document_loaders.notebook",
    "remove_newlines": "langchain_community.document_loaders.notebook",
    "NotebookLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "concatenate_cells",
    "remove_newlines",
    "NotebookLoader",
]
