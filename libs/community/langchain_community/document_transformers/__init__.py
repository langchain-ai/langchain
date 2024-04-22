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
        BeautifulSoupTransformer,  # noqa: F401
    )
    from langchain_community.document_transformers.doctran_text_extract import (
        DoctranPropertyExtractor,  # noqa: F401
    )
    from langchain_community.document_transformers.doctran_text_qa import (
        DoctranQATransformer,  # noqa: F401
    )
    from langchain_community.document_transformers.doctran_text_translate import (
        DoctranTextTranslator,  # noqa: F401
    )
    from langchain_community.document_transformers.embeddings_redundant_filter import (
        EmbeddingsClusteringFilter,  # noqa: F401
        EmbeddingsRedundantFilter,  # noqa: F401
        get_stateful_documents,  # noqa: F401
    )
    from langchain_community.document_transformers.google_translate import (
        GoogleTranslateTransformer,  # noqa: F401
    )
    from langchain_community.document_transformers.html2text import (
        Html2TextTransformer,  # noqa: F401
    )
    from langchain_community.document_transformers.long_context_reorder import (
        LongContextReorder,  # noqa: F401
    )
    from langchain_community.document_transformers.nuclia_text_transform import (
        NucliaTextTransformer,  # noqa: F401
    )
    from langchain_community.document_transformers.openai_functions import (
        OpenAIMetadataTagger,  # noqa: F401
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
