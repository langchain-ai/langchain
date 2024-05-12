"""**Document Transformers** are classes to transform Documents.

**Document Transformers** usually used to transform a lot of Documents in a single run.

**Class hierarchy:**

.. code-block::

    BaseDocumentTransformer --> <name>  # Examples: DoctranQATransformer, DoctranTextTranslator

**Main helpers:**

.. code-block::

    Document
"""  # noqa: E501

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.document_transformers.beautiful_soup_transformer import (
        BeautifulSoupTransformer,
    )
    from langchain_community.document_transformers.doctran_text_extract import (
        DoctranPropertyExtractor,
    )
    from langchain_community.document_transformers.doctran_text_qa import (
        DoctranQATransformer,
    )
    from langchain_community.document_transformers.doctran_text_translate import (
        DoctranTextTranslator,
    )
    from langchain_community.document_transformers.embeddings_redundant_filter import (
        EmbeddingsClusteringFilter,
        EmbeddingsRedundantFilter,
        get_stateful_documents,
    )
    from langchain_community.document_transformers.google_translate import (
        GoogleTranslateTransformer,
    )
    from langchain_community.document_transformers.html2text import (
        Html2TextTransformer,
    )
    from langchain_community.document_transformers.long_context_reorder import (
        LongContextReorder,
    )
    from langchain_community.document_transformers.markdownify import (
        MarkdownifyTransformer,
    )
    from langchain_community.document_transformers.nuclia_text_transform import (
        NucliaTextTransformer,
    )
    from langchain_community.document_transformers.openai_functions import (
        OpenAIMetadataTagger,
    )

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
    "MarkdownifyTransformer",
    "NucliaTextTransformer",
    "OpenAIMetadataTagger",
    "get_stateful_documents",
]

_module_lookup = {
    "BeautifulSoupTransformer": "langchain_community.document_transformers.beautiful_soup_transformer",  # noqa: E501
    "DoctranPropertyExtractor": "langchain_community.document_transformers.doctran_text_extract",  # noqa: E501
    "DoctranQATransformer": "langchain_community.document_transformers.doctran_text_qa",
    "DoctranTextTranslator": "langchain_community.document_transformers.doctran_text_translate",  # noqa: E501
    "EmbeddingsClusteringFilter": "langchain_community.document_transformers.embeddings_redundant_filter",  # noqa: E501
    "EmbeddingsRedundantFilter": "langchain_community.document_transformers.embeddings_redundant_filter",  # noqa: E501
    "GoogleTranslateTransformer": "langchain_community.document_transformers.google_translate",  # noqa: E501
    "Html2TextTransformer": "langchain_community.document_transformers.html2text",
    "LongContextReorder": "langchain_community.document_transformers.long_context_reorder",  # noqa: E501
    "MarkdownifyTransformer": "langchain_community.document_transformers.markdownify",  # noqa: E501
    "NucliaTextTransformer": "langchain_community.document_transformers.nuclia_text_transform",  # noqa: E501
    "OpenAIMetadataTagger": "langchain_community.document_transformers.openai_functions",  # noqa: E501
    "get_stateful_documents": "langchain_community.document_transformers.embeddings_redundant_filter",  # noqa: E501
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
