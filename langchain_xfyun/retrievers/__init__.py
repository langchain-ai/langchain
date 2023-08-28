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

from langchain_xfyun.retrievers.arxiv import ArxivRetriever
from langchain_xfyun.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from langchain_xfyun.retrievers.bm25 import BM25Retriever
from langchain_xfyun.retrievers.chaindesk import ChaindeskRetriever
from langchain_xfyun.retrievers.chatgpt_plugin_retriever import ChatGPTPluginRetriever
from langchain_xfyun.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_xfyun.retrievers.docarray import DocArrayRetriever
from langchain_xfyun.retrievers.elastic_search_bm25 import ElasticSearchBM25Retriever
from langchain_xfyun.retrievers.ensemble import EnsembleRetriever
from langchain_xfyun.retrievers.google_cloud_enterprise_search import (
    GoogleCloudEnterpriseSearchRetriever,
)
from langchain_xfyun.retrievers.kendra import AmazonKendraRetriever
from langchain_xfyun.retrievers.knn import KNNRetriever
from langchain_xfyun.retrievers.llama_index import (
    LlamaIndexGraphRetriever,
    LlamaIndexRetriever,
)
from langchain_xfyun.retrievers.merger_retriever import MergerRetriever
from langchain_xfyun.retrievers.metal import MetalRetriever
from langchain_xfyun.retrievers.milvus import MilvusRetriever
from langchain_xfyun.retrievers.multi_query import MultiQueryRetriever
from langchain_xfyun.retrievers.multi_vector import MultiVectorRetriever
from langchain_xfyun.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_xfyun.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever
from langchain_xfyun.retrievers.pubmed import PubMedRetriever
from langchain_xfyun.retrievers.re_phraser import RePhraseQueryRetriever
from langchain_xfyun.retrievers.remote_retriever import RemoteLangChainRetriever
from langchain_xfyun.retrievers.self_query.base import SelfQueryRetriever
from langchain_xfyun.retrievers.svm import SVMRetriever
from langchain_xfyun.retrievers.tfidf import TFIDFRetriever
from langchain_xfyun.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)
from langchain_xfyun.retrievers.vespa_retriever import VespaRetriever
from langchain_xfyun.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain_xfyun.retrievers.web_research import WebResearchRetriever
from langchain_xfyun.retrievers.wikipedia import WikipediaRetriever
from langchain_xfyun.retrievers.zep import ZepRetriever
from langchain_xfyun.retrievers.zilliz import ZillizRetriever

__all__ = [
    "AmazonKendraRetriever",
    "ArxivRetriever",
    "AzureCognitiveSearchRetriever",
    "ChatGPTPluginRetriever",
    "ContextualCompressionRetriever",
    "ChaindeskRetriever",
    "ElasticSearchBM25Retriever",
    "GoogleCloudEnterpriseSearchRetriever",
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
