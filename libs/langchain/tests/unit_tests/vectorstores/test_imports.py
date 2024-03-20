import pytest

from langchain import vectorstores
from tests.unit_tests import assert_all_importable

EXPECTED_DEPRECATED_IMPORTS = [
    "AlibabaCloudOpenSearch",
    "AlibabaCloudOpenSearchSettings",
    "AnalyticDB",
    "Annoy",
    "AtlasDB",
    "AwaDB",
    "AzureSearch",
    "Bagel",
    "Cassandra",
    "AstraDB",
    "Chroma",
    "Clarifai",
    "Clickhouse",
    "ClickhouseSettings",
    "DashVector",
    "DatabricksVectorSearch",
    "DeepLake",
    "Dingo",
    "DocArrayHnswSearch",
    "DocArrayInMemorySearch",
    "ElasticKnnSearch",
    "ElasticVectorSearch",
    "ElasticsearchStore",
    "Epsilla",
    "FAISS",
    "Hologres",
    "LanceDB",
    "LLMRails",
    "Marqo",
    "MatchingEngine",
    "Meilisearch",
    "Milvus",
    "MomentoVectorIndex",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "Neo4jVector",
    "OpenSearchVectorSearch",
    "PGEmbedding",
    "PGVector",
    "Pinecone",
    "Qdrant",
    "Redis",
    "Rockset",
    "SKLearnVectorStore",
    "ScaNN",
    "SemaDB",
    "SingleStoreDB",
    "SQLiteVSS",
    "StarRocks",
    "SupabaseVectorStore",
    "Tair",
    "TileDB",
    "Tigris",
    "TimescaleVector",
    "Typesense",
    "USearch",
    "Vald",
    "Vearch",
    "Vectara",
    "VespaStore",
    "Weaviate",
    "Yellowbrick",
    "ZepVectorStore",
    "Zilliz",
    "TencentVectorDB",
    "AzureCosmosDBVectorSearch",
]
EXPECTED_ALL = ["VectorStore"]


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(vectorstores, import_)
            assert isinstance(e, ImportError), f"{import_=} didn't raise error"
            assert "langchain_community" in e
    with pytest.raises(AttributeError):
        getattr(vectorstores, "foo")


def test_all_imports() -> None:
    assert set(vectorstores.__all__) == set(EXPECTED_ALL)
    assert_all_importable(vectorstores)
