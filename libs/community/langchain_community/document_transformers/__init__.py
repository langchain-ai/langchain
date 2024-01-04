"""**Document Transformers** are classes to transform Documents.

**Document Transformers** usually used to transform a lot of Documents in a single run.

**Class hierarchy:**

.. code-block::

    BaseDocumentTransformer --> <name>  # Examples: DoctranQATransformer, DoctranTextTranslator

**Main helpers:**

.. code-block::

    Document
"""  # noqa: E501

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
from langchain_community.document_transformers.html2text import Html2TextTransformer
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder,
)
from langchain_community.document_transformers.nuclia_text_transform import (
    NucliaTextTransformer,
)
from langchain_community.document_transformers.openai_functions import (
    OpenAIMetadataTagger,
)

__all__ = [
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
