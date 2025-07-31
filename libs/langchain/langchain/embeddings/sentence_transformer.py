"""Deprecated module for sentence transformer embeddings.

This module previously provided SentenceTransformerEmbeddings which used the
sentence-transformers library. The sentence-transformers dependency has been removed
from LangChain.

For embedding functionality, please use:
- HuggingFaceEmbeddings from langchain_huggingface which now uses transformers directly
- Or install langchain-community which may still contain the legacy implementation

Example migration:
    # Old way (deprecated):
    # from langchain.embeddings import SentenceTransformerEmbeddings
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # New way:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
"""  # noqa: E501

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.embeddings import SentenceTransformerEmbeddings

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {"SentenceTransformerEmbeddings": "langchain_community.embeddings"}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = ["SentenceTransformerEmbeddings"]
