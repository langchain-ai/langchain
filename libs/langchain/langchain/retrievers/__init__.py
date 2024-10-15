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

from typing import TYPE_CHECKING, Any

from langchain._api.module_import import create_importer
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers.re_phraser import RePhraseQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)

if TYPE_CHECKING:
    from langchain_community.retrievers import (
        AmazonKendraRetriever,
        AmazonKnowledgeBasesRetriever,
        ArceeRetriever,
        ArxivRetriever,
        AzureAISearchRetriever,
        AzureCognitiveSearchRetriever,
        BM25Retriever,
        ChaindeskRetriever,
        ChatGPTPluginRetriever,
        CohereRagRetriever,
        DocArrayRetriever,
        DriaRetriever,
        ElasticSearchBM25Retriever,
        EmbedchainRetriever,
        GoogleCloudEnterpriseSearchRetriever,
        GoogleDocumentAIWarehouseRetriever,
        GoogleVertexAIMultiTurnSearchRetriever,
        GoogleVertexAISearchRetriever,
        KayAiRetriever,
        KNNRetriever,
        LlamaIndexGraphRetriever,
        LlamaIndexRetriever,
        MetalRetriever,
        MilvusRetriever,
        NeuralDBRetriever,
        NimbleItRetriever,
        OutlineRetriever,
        PineconeHybridSearchRetriever,
        PubMedRetriever,
        RemoteLangChainRetriever,
        SVMRetriever,
        TavilySearchAPIRetriever,
        TFIDFRetriever,
        VespaRetriever,
        WeaviateHybridSearchRetriever,
        WebResearchRetriever,
        WikipediaRetriever,
        ZepRetriever,
        ZillizRetriever,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "AmazonKendraRetriever": "langchain_community.retrievers",
    "AmazonKnowledgeBasesRetriever": "langchain_community.retrievers",
    "ArceeRetriever": "langchain_community.retrievers",
    "ArxivRetriever": "langchain_community.retrievers",
    "AzureAISearchRetriever": "langchain_community.retrievers",
    "AzureCognitiveSearchRetriever": "langchain_community.retrievers",
    "ChatGPTPluginRetriever": "langchain_community.retrievers",
    "ChaindeskRetriever": "langchain_community.retrievers",
    "CohereRagRetriever": "langchain_community.retrievers",
    "ElasticSearchBM25Retriever": "langchain_community.retrievers",
    "EmbedchainRetriever": "langchain_community.retrievers",
    "GoogleDocumentAIWarehouseRetriever": "langchain_community.retrievers",
    "GoogleCloudEnterpriseSearchRetriever": "langchain_community.retrievers",
    "GoogleVertexAIMultiTurnSearchRetriever": "langchain_community.retrievers",
    "GoogleVertexAISearchRetriever": "langchain_community.retrievers",
    "KayAiRetriever": "langchain_community.retrievers",
    "KNNRetriever": "langchain_community.retrievers",
    "LlamaIndexGraphRetriever": "langchain_community.retrievers",
    "LlamaIndexRetriever": "langchain_community.retrievers",
    "MetalRetriever": "langchain_community.retrievers",
    "MilvusRetriever": "langchain_community.retrievers",
    "NimbleItRetriever":"langchain_community.retrievers",
    "OutlineRetriever": "langchain_community.retrievers",
    "PineconeHybridSearchRetriever": "langchain_community.retrievers",
    "PubMedRetriever": "langchain_community.retrievers",
    "RemoteLangChainRetriever": "langchain_community.retrievers",
    "SVMRetriever": "langchain_community.retrievers",
    "TavilySearchAPIRetriever": "langchain_community.retrievers",
    "BM25Retriever": "langchain_community.retrievers",
    "DriaRetriever": "langchain_community.retrievers",
    "NeuralDBRetriever": "langchain_community.retrievers",
    "TFIDFRetriever": "langchain_community.retrievers",
    "VespaRetriever": "langchain_community.retrievers",
    "WeaviateHybridSearchRetriever": "langchain_community.retrievers",
    "WebResearchRetriever": "langchain_community.retrievers",
    "WikipediaRetriever": "langchain_community.retrievers",
    "ZepRetriever": "langchain_community.retrievers",
    "ZillizRetriever": "langchain_community.retrievers",
    "DocArrayRetriever": "langchain_community.retrievers",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "ArceeRetriever",
    "ArxivRetriever",
    "AzureAISearchRetriever",
    "AzureCognitiveSearchRetriever",
    "BM25Retriever",
    "ChaindeskRetriever",
    "ChatGPTPluginRetriever",
    "CohereRagRetriever",
    "ContextualCompressionRetriever",
    "DocArrayRetriever",
    "DriaRetriever",
    "ElasticSearchBM25Retriever",
    "EmbedchainRetriever",
    "EnsembleRetriever",
    "GoogleCloudEnterpriseSearchRetriever",
    "GoogleDocumentAIWarehouseRetriever",
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
    "MultiVectorRetriever",
    "NimbleItRetriever",
    "OutlineRetriever",
    "ParentDocumentRetriever",
    "PineconeHybridSearchRetriever",
    "PubMedRetriever",
    "RemoteLangChainRetriever",
    "RePhraseQueryRetriever",
    "SelfQueryRetriever",
    "SVMRetriever",
    "TavilySearchAPIRetriever",
    "TFIDFRetriever",
    "TimeWeightedVectorStoreRetriever",
    "VespaRetriever",
    "WeaviateHybridSearchRetriever",
    "WebResearchRetriever",
    "WikipediaRetriever",
    "ZepRetriever",
    "NeuralDBRetriever",
    "ZillizRetriever",
]
