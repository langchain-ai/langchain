"""**Vector store** stores embedded data and performs vector search.

One of the most common ways to store and search over unstructured data is to
embed it and store the resulting embedding vectors, and then query the store
and retrieve the data that are 'most similar' to the embedded query.

**Class hierarchy:**

.. code-block::

    VectorStore --> <name>  # Examples: Annoy, FAISS, Milvus

    BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: VespaRetriever

**Main helpers:**

.. code-block::

    Embeddings, Document
"""  # noqa: E501

import importlib
from typing import Any

_module_lookup = {
    "AlibabaCloudOpenSearch": "langchain_community.vectorstores.alibabacloud_opensearch",  # noqa: E501
    "AlibabaCloudOpenSearchSettings": "langchain_community.vectorstores.alibabacloud_opensearch",  # noqa: E501
    "AnalyticDB": "langchain_community.vectorstores.analyticdb",
    "Annoy": "langchain_community.vectorstores.annoy",
    "ApacheDoris": "langchain_community.vectorstores.apache_doris",
    "AstraDB": "langchain_community.vectorstores.astradb",
    "AtlasDB": "langchain_community.vectorstores.atlas",
    "AwaDB": "langchain_community.vectorstores.awadb",
    "AzureCosmosDBVectorSearch": "langchain_community.vectorstores.azure_cosmos_db",
    "AzureSearch": "langchain_community.vectorstores.azuresearch",
    "BaiduVectorDB": "langchain_community.vectorstores.baiduvectordb",
    "BESVectorStore": "langchain_community.vectorstores.baiducloud_vector_search",
    "Bagel": "langchain_community.vectorstores.bageldb",
    "BigQueryVectorSearch": "langchain_community.vectorstores.bigquery_vector_search",
    "Cassandra": "langchain_community.vectorstores.cassandra",
    "Chroma": "langchain_community.vectorstores.chroma",
    "Clarifai": "langchain_community.vectorstores.clarifai",
    "Clickhouse": "langchain_community.vectorstores.clickhouse",
    "ClickhouseSettings": "langchain_community.vectorstores.clickhouse",
    "CouchbaseVectorStore": "langchain_community.vectorstores.couchbase",
    "DashVector": "langchain_community.vectorstores.dashvector",
    "DatabricksVectorSearch": "langchain_community.vectorstores.databricks_vector_search",  # noqa: E501
    "DeepLake": "langchain_community.vectorstores.deeplake",
    "Dingo": "langchain_community.vectorstores.dingo",
    "DistanceStrategy": "langchain_community.vectorstores.kinetica",
    "DocArrayHnswSearch": "langchain_community.vectorstores.docarray",
    "DocArrayInMemorySearch": "langchain_community.vectorstores.docarray",
    "DocumentDBVectorSearch": "langchain_community.vectorstores.documentdb",
    "ElasticKnnSearch": "langchain_community.vectorstores.elastic_vector_search",
    "ElasticVectorSearch": "langchain_community.vectorstores.elastic_vector_search",
    "ElasticsearchStore": "langchain_community.vectorstores.elasticsearch",
    "Epsilla": "langchain_community.vectorstores.epsilla",
    "FAISS": "langchain_community.vectorstores.faiss",
    "HanaDB": "langchain_community.vectorstores.hanavector",
    "Hologres": "langchain_community.vectorstores.hologres",
    "InfinispanVS": "langchain_community.vectorstores.infinispanvs",
    "KDBAI": "langchain_community.vectorstores.kdbai",
    "Kinetica": "langchain_community.vectorstores.kinetica",
    "KineticaSettings": "langchain_community.vectorstores.kinetica",
    "LLMRails": "langchain_community.vectorstores.llm_rails",
    "LanceDB": "langchain_community.vectorstores.lancedb",
    "Lantern": "langchain_community.vectorstores.lantern",
    "Marqo": "langchain_community.vectorstores.marqo",
    "MatchingEngine": "langchain_community.vectorstores.matching_engine",
    "Meilisearch": "langchain_community.vectorstores.meilisearch",
    "Milvus": "langchain_community.vectorstores.milvus",
    "MomentoVectorIndex": "langchain_community.vectorstores.momento_vector_index",
    "MongoDBAtlasVectorSearch": "langchain_community.vectorstores.mongodb_atlas",
    "MyScale": "langchain_community.vectorstores.myscale",
    "MyScaleSettings": "langchain_community.vectorstores.myscale",
    "Neo4jVector": "langchain_community.vectorstores.neo4j_vector",
    "NeuralDBVectorStore": "langchain_community.vectorstores.thirdai_neuraldb",
    "OpenSearchVectorSearch": "langchain_community.vectorstores.opensearch_vector_search",  # noqa: E501
    "PGEmbedding": "langchain_community.vectorstores.pgembedding",
    "PGVector": "langchain_community.vectorstores.pgvector",
    "Pinecone": "langchain_community.vectorstores.pinecone",
    "Qdrant": "langchain_community.vectorstores.qdrant",
    "Redis": "langchain_community.vectorstores.redis",
    "Rockset": "langchain_community.vectorstores.rocksetdb",
    "SKLearnVectorStore": "langchain_community.vectorstores.sklearn",
    "SQLiteVSS": "langchain_community.vectorstores.sqlitevss",
    "ScaNN": "langchain_community.vectorstores.scann",
    "SemaDB": "langchain_community.vectorstores.semadb",
    "SingleStoreDB": "langchain_community.vectorstores.singlestoredb",
    "StarRocks": "langchain_community.vectorstores.starrocks",
    "SupabaseVectorStore": "langchain_community.vectorstores.supabase",
    "SurrealDBStore": "langchain_community.vectorstores.surrealdb",
    "Tair": "langchain_community.vectorstores.tair",
    "TencentVectorDB": "langchain_community.vectorstores.tencentvectordb",
    "TiDBVectorStore": "langchain_community.vectorstores.tidb_vector",
    "Tigris": "langchain_community.vectorstores.tigris",
    "TileDB": "langchain_community.vectorstores.tiledb",
    "TimescaleVector": "langchain_community.vectorstores.timescalevector",
    "Typesense": "langchain_community.vectorstores.typesense",
    "USearch": "langchain_community.vectorstores.usearch",
    "Vald": "langchain_community.vectorstores.vald",
    "Vearch": "langchain_community.vectorstores.vearch",
    "Vectara": "langchain_community.vectorstores.vectara",
    "VectorStore": "langchain_core.vectorstores",
    "VespaStore": "langchain_community.vectorstores.vespa",
    "Weaviate": "langchain_community.vectorstores.weaviate",
    "Yellowbrick": "langchain_community.vectorstores.yellowbrick",
    "ZepVectorStore": "langchain_community.vectorstores.zep",
    "Zilliz": "langchain_community.vectorstores.zilliz",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
