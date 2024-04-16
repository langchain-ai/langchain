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

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.retrievers.arcee import (
        ArceeRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.arxiv import (
        ArxivRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.azure_cognitive_search import (
        AzureCognitiveSearchRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.bedrock import (
        AmazonKnowledgeBasesRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.bm25 import (
        BM25Retriever,  # noqa: F401
    )
    from langchain_community.retrievers.breebs import (
        BreebsRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.chaindesk import (
        ChaindeskRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.chatgpt_plugin_retriever import (
        ChatGPTPluginRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.cohere_rag_retriever import (
        CohereRagRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.docarray import (
        DocArrayRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.dria_index import (
        DriaRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.elastic_search_bm25 import (
        ElasticSearchBM25Retriever,  # noqa: F401
    )
    from langchain_community.retrievers.embedchain import (
        EmbedchainRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.google_cloud_documentai_warehouse import (
        GoogleDocumentAIWarehouseRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.google_vertex_ai_search import (
        GoogleCloudEnterpriseSearchRetriever,  # noqa: F401
        GoogleVertexAIMultiTurnSearchRetriever,  # noqa: F401
        GoogleVertexAISearchRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.kay import (
        KayAiRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.kendra import (
        AmazonKendraRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.knn import (
        KNNRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.llama_index import (
        LlamaIndexGraphRetriever,  # noqa: F401
        LlamaIndexRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.metal import (
        MetalRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.milvus import (
        MilvusRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.outline import (
        OutlineRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.pinecone_hybrid_search import (
        PineconeHybridSearchRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.pubmed import (
        PubMedRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.qdrant_sparse_vector_retriever import (
        QdrantSparseVectorRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.remote_retriever import (
        RemoteLangChainRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.svm import (
        SVMRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.tavily_search_api import (
        TavilySearchAPIRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.tfidf import (
        TFIDFRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.vespa_retriever import (
        VespaRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.weaviate_hybrid_search import (
        WeaviateHybridSearchRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.wikipedia import (
        WikipediaRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.you import (
        YouRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.zep import (
        ZepRetriever,  # noqa: F401
    )
    from langchain_community.retrievers.zilliz import (
        ZillizRetriever,  # noqa: F401
    )

__all__ = [
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "ArceeRetriever",
    "ArxivRetriever",
    "AzureCognitiveSearchRetriever",
    "BM25Retriever",
    "BreebsRetriever",
    "ChaindeskRetriever",
    "ChatGPTPluginRetriever",
    "CohereRagRetriever",
    "DocArrayRetriever",
    "DriaRetriever",
    "ElasticSearchBM25Retriever",
    "EmbedchainRetriever",
    "GoogleCloudEnterpriseSearchRetriever",
    "GoogleDocumentAIWarehouseRetriever",
    "GoogleVertexAIMultiTurnSearchRetriever",
    "GoogleVertexAISearchRetriever",
    "KNNRetriever",
    "KayAiRetriever",
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
    "TFIDFRetriever",
    "TavilySearchAPIRetriever",
    "VespaRetriever",
    "WeaviateHybridSearchRetriever",
    "WikipediaRetriever",
    "YouRetriever",
    "ZepRetriever",
    "ZillizRetriever",
]

_module_lookup = {
    "AmazonKendraRetriever": "langchain_community.retrievers.kendra",
    "AmazonKnowledgeBasesRetriever": "langchain_community.retrievers.bedrock",
    "ArceeRetriever": "langchain_community.retrievers.arcee",
    "ArxivRetriever": "langchain_community.retrievers.arxiv",
    "AzureAISearchRetriever": "langchain_community.retrievers.azure_ai_search",  # noqa: E501
    "AzureCognitiveSearchRetriever": "langchain_community.retrievers.azure_ai_search",  # noqa: E501
    "BM25Retriever": "langchain_community.retrievers.bm25",
    "BreebsRetriever": "langchain_community.retrievers.breebs",
    "ChaindeskRetriever": "langchain_community.retrievers.chaindesk",
    "ChatGPTPluginRetriever": "langchain_community.retrievers.chatgpt_plugin_retriever",
    "CohereRagRetriever": "langchain_community.retrievers.cohere_rag_retriever",
    "DocArrayRetriever": "langchain_community.retrievers.docarray",
    "DriaRetriever": "langchain_community.retrievers.dria_index",
    "ElasticSearchBM25Retriever": "langchain_community.retrievers.elastic_search_bm25",
    "EmbedchainRetriever": "langchain_community.retrievers.embedchain",
    "GoogleCloudEnterpriseSearchRetriever": "langchain_community.retrievers.google_vertex_ai_search",  # noqa: E501
    "GoogleDocumentAIWarehouseRetriever": "langchain_community.retrievers.google_cloud_documentai_warehouse",  # noqa: E501
    "GoogleVertexAIMultiTurnSearchRetriever": "langchain_community.retrievers.google_vertex_ai_search",  # noqa: E501
    "GoogleVertexAISearchRetriever": "langchain_community.retrievers.google_vertex_ai_search",  # noqa: E501
    "KNNRetriever": "langchain_community.retrievers.knn",
    "KayAiRetriever": "langchain_community.retrievers.kay",
    "LlamaIndexGraphRetriever": "langchain_community.retrievers.llama_index",
    "LlamaIndexRetriever": "langchain_community.retrievers.llama_index",
    "MetalRetriever": "langchain_community.retrievers.metal",
    "MilvusRetriever": "langchain_community.retrievers.milvus",
    "OutlineRetriever": "langchain_community.retrievers.outline",
    "PineconeHybridSearchRetriever": "langchain_community.retrievers.pinecone_hybrid_search",  # noqa: E501
    "PubMedRetriever": "langchain_community.retrievers.pubmed",
    "QdrantSparseVectorRetriever": "langchain_community.retrievers.qdrant_sparse_vector_retriever",  # noqa: E501
    "RemoteLangChainRetriever": "langchain_community.retrievers.remote_retriever",
    "SVMRetriever": "langchain_community.retrievers.svm",
    "TFIDFRetriever": "langchain_community.retrievers.tfidf",
    "TavilySearchAPIRetriever": "langchain_community.retrievers.tavily_search_api",
    "VespaRetriever": "langchain_community.retrievers.vespa_retriever",
    "WeaviateHybridSearchRetriever": "langchain_community.retrievers.weaviate_hybrid_search",  # noqa: E501
    "WikipediaRetriever": "langchain_community.retrievers.wikipedia",
    "YouRetriever": "langchain_community.retrievers.you",
    "ZepRetriever": "langchain_community.retrievers.zep",
    "ZillizRetriever": "langchain_community.retrievers.zilliz",
    "NeuralDBRetriever": "langchain_community.retrievers.thirdai_neuraldb",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
