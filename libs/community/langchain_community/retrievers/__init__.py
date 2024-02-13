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

from langchain_community.retrievers.arcee import ArceeRetriever
from langchain_community.retrievers.arxiv import ArxivRetriever
from langchain_community.retrievers.azure_cognitive_search import (
    AzureCognitiveSearchRetriever,
)
from langchain_community.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.retrievers.breebs import BreebsRetriever
from langchain_community.retrievers.chaindesk import ChaindeskRetriever
from langchain_community.retrievers.chatgpt_plugin_retriever import (
    ChatGPTPluginRetriever,
)
from langchain_community.retrievers.cohere_rag_retriever import CohereRagRetriever
from langchain_community.retrievers.docarray import DocArrayRetriever
from langchain_community.retrievers.elastic_search_bm25 import (
    ElasticSearchBM25Retriever,
)
from langchain_community.retrievers.embedchain import EmbedchainRetriever
from langchain_community.retrievers.google_cloud_documentai_warehouse import (
    GoogleDocumentAIWarehouseRetriever,
)
from langchain_community.retrievers.google_vertex_ai_search import (
    GoogleCloudEnterpriseSearchRetriever,
    GoogleVertexAIMultiTurnSearchRetriever,
    GoogleVertexAISearchRetriever,
)
from langchain_community.retrievers.kay import KayAiRetriever
from langchain_community.retrievers.kendra import AmazonKendraRetriever
from langchain_community.retrievers.knn import KNNRetriever
from langchain_community.retrievers.llama_index import (
    LlamaIndexGraphRetriever,
    LlamaIndexRetriever,
)
from langchain_community.retrievers.metal import MetalRetriever
from langchain_community.retrievers.milvus import MilvusRetriever
from langchain_community.retrievers.outline import OutlineRetriever
from langchain_community.retrievers.pinecone_hybrid_search import (
    PineconeHybridSearchRetriever,
)
from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain_community.retrievers.qdrant_sparse_vector_retriever import (
    QdrantSparseVectorRetriever,
)
from langchain_community.retrievers.remote_retriever import RemoteLangChainRetriever
from langchain_community.retrievers.svm import SVMRetriever
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_community.retrievers.tfidf import TFIDFRetriever
from langchain_community.retrievers.vespa_retriever import VespaRetriever
from langchain_community.retrievers.weaviate_hybrid_search import (
    WeaviateHybridSearchRetriever,
)
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_community.retrievers.you import YouRetriever
from langchain_community.retrievers.zep import ZepRetriever
from langchain_community.retrievers.zilliz import ZillizRetriever

__all__ = [
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "ArceeRetriever",
    "ArxivRetriever",
    "AzureCognitiveSearchRetriever",
    "BM25Retriever",
    "BreebsRetriever",
    "ChatGPTPluginRetriever",
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
    "MetalRetriever",
    "MilvusRetriever",
    "OutlineRetriever",
    "PineconeHybridSearchRetriever",
    "PubMedRetriever",
    "QdrantSparseVectorRetriever",
    "RemoteLangChainRetriever",
    "SVMRetriever",
    "TavilySearchAPIRetriever",
    "TFIDFRetriever",
    "VespaRetriever",
    "WeaviateHybridSearchRetriever",
    "WikipediaRetriever",
    "YouRetriever",
    "ZepRetriever",
    "ZillizRetriever",
    "DocArrayRetriever",
]
