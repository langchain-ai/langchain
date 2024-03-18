"""DEPRECATED: This module has been moved to the langchain-community package.

**Document Transformers** are classes to transform Documents.

"""
from typing import Any

DEPRECATED_IMPORTS = [
    "BeautifulSoupTransformer",
    "DoctranQATransformer",
    "DoctranTextTranslator",
    "DoctranPropertyExtractor",
    "EmbeddingsClusteringFilter",
    "EmbeddingsRedundantFilter",
    "GoogleTranslateTransformer",
    "get_stateful_documents",
    "LongContextReorder",
    "NucliaTextTransformer",
    "OpenAIMetadataTagger",
    "Html2TextTransformer",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.document_transformers import {name}`"
        )
    raise AttributeError()
