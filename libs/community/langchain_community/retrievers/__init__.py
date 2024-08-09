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
        ArceeRetriever,
    )
    from langchain_community.retrievers.arxiv import (
        ArxivRetriever,
    )
    from langchain_community.retrievers.asknews import (
        AskNewsRetriever,
    )
    from langchain_community.retrievers.azure_ai_search import (
        AzureAISearchRetriever,
        AzureCognitiveSearchRetriever,
    )
    from langchain_community.retrievers.bedrock import (
        AmazonKnowledgeBasesRetriever,
    )
    from langchain_community.retrievers.bm25 import (
        BM25Retriever,
    )
    from langchain_community.retrievers.bm25s import (
        BM25SRetriever,
    )
    from langchain_community.retrievers.breebs import (
        BreebsRetriever,
    )
    from langchain_community.retrievers.chaindesk import (
        ChaindeskRetriever,
    )
    from langchain_community.retrievers.chatgpt_plugin_retriever import (
        ChatGPTPluginRetriever,
    )
    from langchain_community.retrievers.cohere_rag_retriever import (
        CohereRagRetriever,
    )
    from langchain_community.retrievers.docarray import (
        DocArrayRetriever,
    )
    from langchain_community.retrievers.dria_index import (
        DriaRetriever,
    )
    from langchain_community.retrievers.elastic_search_bm25 import (
        ElasticSearchBM25Retriever,
    )
    from langchain_community.retrievers.embedchain import (
        EmbedchainRetriever,
    )
    from langchain_community.retrievers.google_cloud_documentai_warehouse import (
        GoogleDocumentAIWarehouseRetriever,
    )
    from langchain_community.retrievers.google_vertex_ai_search import (
        GoogleCloudEnterpriseSearchRetriever,
        GoogleVertexAIMultiTurnSearchRetriever,
        GoogleVertexAISearchRetriever,
    )
    from langchain_community.retrievers.kay import (
        KayAiRetriever,
    )
    from langchain_community.retrievers.kendra import (
        AmazonKendraRetriever,
    )
    from langchain_community.retrievers.knn import (
        KNNRetriever,
    )
    from langchain_community.retrievers.llama_index import (
        LlamaIndexGraphRetriever,
        LlamaIndexRetriever,
    )
    from langchain_community.retrievers.metal import (
        MetalRetriever,
    )
    from langchain_community.retrievers.milvus import (
        MilvusRetriever,
    )
    from langchain_community.retrievers.nanopq import NanoPQRetriever
    from langchain_community.retrievers.outline import (
        OutlineRetriever,
    )
    from langchain_community.retrievers.pinecone_hybrid_search import (
        PineconeHybridSearchRetriever,
    )
    from langchain_community.retrievers.pubmed import (
        PubMedRetriever,
    )
    from langchain_community.retrievers.qdrant_sparse_vector_retriever import (
        QdrantSparseVectorRetriever,
    )
    from langchain_community.retrievers.rememberizer import (
        RememberizerRetriever,
    )
    from langchain_community.retrievers.remote_retriever import (
        RemoteLangChainRetriever,
    )
    from langchain_community.retrievers.svm import (
        SVMRetriever,
    )
    from langchain_community.retrievers.tavily_search_api import (
        TavilySearchAPIRetriever,
    )
    from langchain_community.retrievers.tfidf import (
        TFIDFRetriever,
    )
    from langchain_community.retrievers.thirdai_neuraldb import NeuralDBRetriever
    from langchain_community.retrievers.vespa_retriever import (
        VespaRetriever,
    )
    from langchain_community.retrievers.weaviate_hybrid_search import (
        WeaviateHybridSearchRetriever,
    )
    from langchain_community.retrievers.web_research import WebResearchRetriever
    from langchain_community.retrievers.wikipedia import (
        WikipediaRetriever,
    )
    from langchain_community.retrievers.you import (
        YouRetriever,
    )
    from langchain_community.retrievers.zep import (
        ZepRetriever,
    )
    from langchain_community.retrievers.zep_cloud import (
        ZepCloudRetriever,
    )
    from langchain_community.retrievers.zilliz import (
        ZillizRetriever,
    )


_module_lookup = {
    "AmazonKendraRetriever": "langchain_community.retrievers.kendra",
    "AmazonKnowledgeBasesRetriever": "langchain_community.retrievers.bedrock",
    "ArceeRetriever": "langchain_community.retrievers.arcee",
    "ArxivRetriever": "langchain_community.retrievers.arxiv",
    "AskNewsRetriever": "langchain_community.retrievers.asknews",
    "AzureAISearchRetriever": "langchain_community.retrievers.azure_ai_search",
    "AzureCognitiveSearchRetriever": "langchain_community.retrievers.azure_ai_search",
    "BM25Retriever": "langchain_community.retrievers.bm25",
    "BM25SRetriever": "langchain_community.retrievers.bm25s",
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
    "NanoPQRetriever": "langchain_community.retrievers.nanopq",
    "OutlineRetriever": "langchain_community.retrievers.outline",
    "PineconeHybridSearchRetriever": "langchain_community.retrievers.pinecone_hybrid_search",  # noqa: E501
    "PubMedRetriever": "langchain_community.retrievers.pubmed",
    "QdrantSparseVectorRetriever": "langchain_community.retrievers.qdrant_sparse_vector_retriever",  # noqa: E501
    "RememberizerRetriever": "langchain_community.retrievers.rememberizer",
    "RemoteLangChainRetriever": "langchain_community.retrievers.remote_retriever",
    "SVMRetriever": "langchain_community.retrievers.svm",
    "TFIDFRetriever": "langchain_community.retrievers.tfidf",
    "TavilySearchAPIRetriever": "langchain_community.retrievers.tavily_search_api",
    "VespaRetriever": "langchain_community.retrievers.vespa_retriever",
    "WeaviateHybridSearchRetriever": "langchain_community.retrievers.weaviate_hybrid_search",  # noqa: E501
    "WebResearchRetriever": "langchain_community.retrievers.web_research",
    "WikipediaRetriever": "langchain_community.retrievers.wikipedia",
    "YouRetriever": "langchain_community.retrievers.you",
    "ZepRetriever": "langchain_community.retrievers.zep",
    "ZepCloudRetriever": "langchain_community.retrievers.zep_cloud",
    "ZillizRetriever": "langchain_community.retrievers.zilliz",
    "NeuralDBRetriever": "langchain_community.retrievers.thirdai_neuraldb",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "ArceeRetriever",
    "ArxivRetriever",
    "AskNewsRetriever",
    "AzureAISearchRetriever",
    "AzureCognitiveSearchRetriever",
    "BM25Retriever",
    "BM25SRetriever",
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
    "KayAiRetriever",
    "KNNRetriever",
    "LlamaIndexGraphRetriever",
    "LlamaIndexRetriever",
    "MetalRetriever",
    "MilvusRetriever",
    "NanoPQRetriever",
    "NeuralDBRetriever",
    "OutlineRetriever",
    "PineconeHybridSearchRetriever",
    "PubMedRetriever",
    "QdrantSparseVectorRetriever",
    "RememberizerRetriever",
    "RemoteLangChainRetriever",
    "SVMRetriever",
    "TavilySearchAPIRetriever",
    "TFIDFRetriever",
    "VespaRetriever",
    "WeaviateHybridSearchRetriever",
    "WebResearchRetriever",
    "WikipediaRetriever",
    "YouRetriever",
    "ZepRetriever",
    "ZepCloudRetriever",
    "ZillizRetriever",
]
