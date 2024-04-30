from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.word_document import (
        Docx2txtLoader,
        UnstructuredWordDocumentLoader,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "Docx2txtLoader": "langchain_community.document_loaders.word_document",
    "UnstructuredWordDocumentLoader": (
        "langchain_community.document_loaders.word_document"
    ),
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "Docx2txtLoader",
    "UnstructuredWordDocumentLoader",
]
