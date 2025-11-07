"""**Document Transformers** are classes to transform Documents.

**Document Transformers** usually used to transform a lot of Documents in a single run.
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_transformers import (
        BeautifulSoupTransformer,
        DoctranPropertyExtractor,
        DoctranQATransformer,
        DoctranTextTranslator,
        EmbeddingsClusteringFilter,
        EmbeddingsRedundantFilter,
        GoogleTranslateTransformer,
        Html2TextTransformer,
        LongContextReorder,
        NucliaTextTransformer,
        OpenAIMetadataTagger,
        get_stateful_documents,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "BeautifulSoupTransformer": "langchain_community.document_transformers",
    "DoctranQATransformer": "langchain_community.document_transformers",
    "DoctranTextTranslator": "langchain_community.document_transformers",
    "DoctranPropertyExtractor": "langchain_community.document_transformers",
    "EmbeddingsClusteringFilter": "langchain_community.document_transformers",
    "EmbeddingsRedundantFilter": "langchain_community.document_transformers",
    "GoogleTranslateTransformer": "langchain_community.document_transformers",
    "get_stateful_documents": "langchain_community.document_transformers",
    "LongContextReorder": "langchain_community.document_transformers",
    "NucliaTextTransformer": "langchain_community.document_transformers",
    "OpenAIMetadataTagger": "langchain_community.document_transformers",
    "Html2TextTransformer": "langchain_community.document_transformers",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BeautifulSoupTransformer",
    "DoctranPropertyExtractor",
    "DoctranQATransformer",
    "DoctranTextTranslator",
    "EmbeddingsClusteringFilter",
    "EmbeddingsRedundantFilter",
    "GoogleTranslateTransformer",
    "Html2TextTransformer",
    "LongContextReorder",
    "NucliaTextTransformer",
    "OpenAIMetadataTagger",
    "get_stateful_documents",
]
