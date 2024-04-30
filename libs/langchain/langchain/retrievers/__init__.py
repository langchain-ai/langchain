"""**Retriever** class returns Documents given a text **query**.

It is more general than a vector store. A retriever does not need to be able to
store documents, only to return (or retrieve) it. Vector stores can be used as
the backbone of a retriever, but there are other types of retrievers as well.

**Class hierarchy:**

.. code-block::

    BaseRetriever --> <name>Retriever  # Examples: ArxivRetriever, MergerRetriever

**Main helpers:**

.. code-block::

    Document, Serializable, Callbacks,
    CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
"""
from typing import Any

from langchain._api.module_import import create_importer
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.outline import OutlineRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers.re_phraser import RePhraseQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)
from langchain.retrievers.web_research import WebResearchRetriever

import_lookup = create_importer(
    __package__, fallback_module="langchain_community.retrievers"
)


def __getattr__(name: str) -> Any:
    """Import retrievers from langchain_community."""
    return import_lookup(name)


__all__ = [
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "ArceeRetriever",
    "ArxivRetriever",
    "AzureAISearchRetriever",
    "AzureCognitiveSearchRetriever",
    "ChatGPTPluginRetriever",
    "ContextualCompressionRetriever",
    "ChaindeskRetriever",
    "CohereRagRetriever",
    "ElasticSearchBM25Retriever",
    "EmbedchainRetriever",
    "GoogleDocumentAIWarehouseRetriever",
    "GoogleCloudEnterpriseSearchRetriever",
    "GoogleVertexAIMultiTurnSearchRetriever",
    "GoogleVertexAISearchRetriever",
    "KayAiRetriever",
    "KNNRetriever",
    "LlamaIndexGraphRetriever",
    "LlamaIndexRetriever",
    "MergerRetriever",
    "MetalRetriever",
    "MilvusRetriever",
    "MultiQueryRetriever",
    "OutlineRetriever",
    "PineconeHybridSearchRetriever",
    "PubMedRetriever",
    "RemoteLangChainRetriever",
    "SVMRetriever",
    "SelfQueryRetriever",
    "TavilySearchAPIRetriever",
    "TFIDFRetriever",
    "BM25Retriever",
    "TimeWeightedVectorStoreRetriever",
    "VespaRetriever",
    "WeaviateHybridSearchRetriever",
    "WikipediaRetriever",
    "ZepRetriever",
    "ZillizRetriever",
    "DocArrayRetriever",
    "RePhraseQueryRetriever",
    "WebResearchRetriever",
    "EnsembleRetriever",
    "ParentDocumentRetriever",
    "MultiVectorRetriever",
]
