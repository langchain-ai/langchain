from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.embeddings import (
        HuggingFaceBgeEmbeddings,
        HuggingFaceEmbeddings,
        HuggingFaceInferenceAPIEmbeddings,
        HuggingFaceInstructEmbeddings,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "HuggingFaceEmbeddings": "langchain_community.embeddings",
    "HuggingFaceInstructEmbeddings": "langchain_community.embeddings",
    "HuggingFaceBgeEmbeddings": "langchain_community.embeddings",
    "HuggingFaceInferenceAPIEmbeddings": "langchain_community.embeddings",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "HuggingFaceBgeEmbeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "HuggingFaceInstructEmbeddings",
]
