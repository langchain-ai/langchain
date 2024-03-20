import pytest

from langchain import retrievers
from tests.unit_tests import assert_all_importable

EXPECTED_DEPRECATED_IMPORTS = [
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "ArceeRetriever",
    "ArxivRetriever",
    "AzureCognitiveSearchRetriever",
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
    "SVMRetriever",
    "TavilySearchAPIRetriever",
    "TFIDFRetriever",
    "BM25Retriever",
    "VespaRetriever",
    "WeaviateHybridSearchRetriever",
    "WikipediaRetriever",
    "ZepRetriever",
    "ZillizRetriever",
    "DocArrayRetriever",
    "RemoteLangChainRetriever",
]

EXPECTED_ALL = [
    "WebResearchRetriever",
    "MergerRetriever",
    "MultiQueryRetriever",
    "SelfQueryRetriever",
    "TimeWeightedVectorStoreRetriever",
    "RePhraseQueryRetriever",
    "EnsembleRetriever",
    "ParentDocumentRetriever",
    "MultiVectorRetriever",
    "ContextualCompressionRetriever",
]


def test_imports() -> None:
    assert sorted(retrievers.__all__) == sorted(EXPECTED_ALL)
    assert_all_importable(retrievers)


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(retrievers, import_)
            assert "langchain_community" in e, f"{import_=} didn't error"
    with pytest.raises(AttributeError):
        getattr(retrievers, "foo")
