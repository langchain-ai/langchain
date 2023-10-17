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

from langchain.retrievers.arcee import ArceeRetriever
from langchain.retrievers.arxiv import ArxivRetriever
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers.chaindesk import ChaindeskRetriever
from langchain.retrievers.chatgpt_plugin_retriever import ChatGPTPluginRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.docarray import DocArrayRetriever
from langchain.retrievers.elastic_search_bm25 import ElasticSearchBM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.google_cloud_documentai_warehouse import (
    GoogleDocumentAIWarehouseRetriever,
)
from langchain.retrievers.google_cloud_enterprise_search import (
    GoogleCloudEnterpriseSearchRetriever,
)
from langchain.retrievers.google_vertex_ai_search import (
    GoogleVertexAIMultiTurnSearchRetriever,
    GoogleVertexAISearchRetriever,
)
from langchain.retrievers.kay import KayAiRetriever
from langchain.retrievers.kendra import AmazonKendraRetriever
from langchain.retrievers.knn import KNNRetriever
from langchain.retrievers.llama_index import (
    LlamaIndexGraphRetriever,
    LlamaIndexRetriever,
)
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.metal import MetalRetriever
from langchain.retrievers.milvus import MilvusRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever
from langchain.retrievers.pubmed import PubMedRetriever
from langchain.retrievers.re_phraser import RePhraseQueryRetriever
from langchain.retrievers.remote_retriever import RemoteLangChainRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.svm import SVMRetriever
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain.retrievers.tfidf import TFIDFRetriever
from langchain.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)
from langchain.retrievers.vespa_retriever import VespaRetriever
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.retrievers.wikipedia import WikipediaRetriever
from langchain.retrievers.zep import ZepRetriever
from langchain.retrievers.zilliz import ZillizRetriever

__all__ = [
    "AmazonKendraRetriever",
    "ArceeRetriever",
    "ArxivRetriever",
    "AzureCognitiveSearchRetriever",
    "ChatGPTPluginRetriever",
    "ContextualCompressionRetriever",
    "ChaindeskRetriever",
    "ElasticSearchBM25Retriever",
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
