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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.vectorstores import (
        VectorStore,
    )

    from langchain_community.vectorstores.aerospike import (
        Aerospike,
    )
    from langchain_community.vectorstores.alibabacloud_opensearch import (
        AlibabaCloudOpenSearch,
        AlibabaCloudOpenSearchSettings,
    )
    from langchain_community.vectorstores.analyticdb import (
        AnalyticDB,
    )
    from langchain_community.vectorstores.annoy import (
        Annoy,
    )
    from langchain_community.vectorstores.apache_doris import (
        ApacheDoris,
    )
    from langchain_community.vectorstores.aperturedb import (
        ApertureDB,
    )
    from langchain_community.vectorstores.astradb import (
        AstraDB,
    )
    from langchain_community.vectorstores.atlas import (
        AtlasDB,
    )
    from langchain_community.vectorstores.awadb import (
        AwaDB,
    )
    from langchain_community.vectorstores.azure_cosmos_db import (
        AzureCosmosDBVectorSearch,
    )
    from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
        AzureCosmosDBNoSqlVectorSearch,
    )
    from langchain_community.vectorstores.azuresearch import (
        AzureSearch,
    )
    from langchain_community.vectorstores.bagel import (
        Bagel,
    )
    from langchain_community.vectorstores.baiducloud_vector_search import (
        BESVectorStore,
    )
    from langchain_community.vectorstores.baiduvectordb import (
        BaiduVectorDB,
    )
    from langchain_community.vectorstores.bigquery_vector_search import (
        BigQueryVectorSearch,
    )
    from langchain_community.vectorstores.cassandra import (
        Cassandra,
    )
    from langchain_community.vectorstores.chroma import (
        Chroma,
    )
    from langchain_community.vectorstores.clarifai import (
        Clarifai,
    )
    from langchain_community.vectorstores.clickhouse import (
        Clickhouse,
        ClickhouseSettings,
    )
    from langchain_community.vectorstores.couchbase import (
        CouchbaseVectorStore,
    )
    from langchain_community.vectorstores.dashvector import (
        DashVector,
    )
    from langchain_community.vectorstores.databricks_vector_search import (
        DatabricksVectorSearch,
    )
    from langchain_community.vectorstores.deeplake import (
        DeepLake,
    )
    from langchain_community.vectorstores.dingo import (
        Dingo,
    )
    from langchain_community.vectorstores.docarray import (
        DocArrayHnswSearch,
        DocArrayInMemorySearch,
    )
    from langchain_community.vectorstores.documentdb import (
        DocumentDBVectorSearch,
    )
    from langchain_community.vectorstores.duckdb import (
        DuckDB,
    )
    from langchain_community.vectorstores.ecloud_vector_search import (
        EcloudESVectorStore,
    )
    from langchain_community.vectorstores.elastic_vector_search import (
        ElasticKnnSearch,
        ElasticVectorSearch,
    )
    from langchain_community.vectorstores.elasticsearch import (
        ElasticsearchStore,
    )
    from langchain_community.vectorstores.epsilla import (
        Epsilla,
    )
    from langchain_community.vectorstores.faiss import (
        FAISS,
    )
    from langchain_community.vectorstores.hanavector import (
        HanaDB,
    )
    from langchain_community.vectorstores.hologres import (
        Hologres,
    )
    from langchain_community.vectorstores.infinispanvs import (
        InfinispanVS,
    )
    from langchain_community.vectorstores.inmemory import (
        InMemoryVectorStore,
    )
    from langchain_community.vectorstores.kdbai import (
        KDBAI,
    )
    from langchain_community.vectorstores.kinetica import (
        DistanceStrategy,
        Kinetica,
        KineticaSettings,
    )
    from langchain_community.vectorstores.lancedb import (
        LanceDB,
    )
    from langchain_community.vectorstores.lantern import (
        Lantern,
    )
    from langchain_community.vectorstores.llm_rails import (
        LLMRails,
    )
    from langchain_community.vectorstores.manticore_search import (
        ManticoreSearch,
        ManticoreSearchSettings,
    )
    from langchain_community.vectorstores.marqo import (
        Marqo,
    )
    from langchain_community.vectorstores.matching_engine import (
        MatchingEngine,
    )
    from langchain_community.vectorstores.meilisearch import (
        Meilisearch,
    )
    from langchain_community.vectorstores.milvus import (
        Milvus,
    )
    from langchain_community.vectorstores.momento_vector_index import (
        MomentoVectorIndex,
    )
    from langchain_community.vectorstores.mongodb_atlas import (
        MongoDBAtlasVectorSearch,
    )
    from langchain_community.vectorstores.myscale import (
        MyScale,
        MyScaleSettings,
    )
    from langchain_community.vectorstores.neo4j_vector import (
        Neo4jVector,
    )
    from langchain_community.vectorstores.oceanbase import (
        OceanBase,
    )
    from langchain_community.vectorstores.opensearch_vector_search import (
        OpenSearchVectorSearch,
    )
    from langchain_community.vectorstores.oraclevs import (
        OracleVS,
    )
    from langchain_community.vectorstores.pathway import (
        PathwayVectorClient,
    )
    from langchain_community.vectorstores.pgembedding import (
        PGEmbedding,
    )
    from langchain_community.vectorstores.pgvector import (
        PGVector,
    )
    from langchain_community.vectorstores.pinecone import (
        Pinecone,
    )
    from langchain_community.vectorstores.qdrant import (
        Qdrant,
    )
    from langchain_community.vectorstores.redis import (
        Redis,
    )
    from langchain_community.vectorstores.relyt import (
        Relyt,
    )
    from langchain_community.vectorstores.rocksetdb import (
        Rockset,
    )
    from langchain_community.vectorstores.scann import (
        ScaNN,
    )
    from langchain_community.vectorstores.semadb import (
        SemaDB,
    )
    from langchain_community.vectorstores.singlestoredb import (
        SingleStoreDB,
    )
    from langchain_community.vectorstores.sklearn import (
        SKLearnVectorStore,
    )
    from langchain_community.vectorstores.sqlitevec import (
        SQLiteVec,
    )
    from langchain_community.vectorstores.sqlitevss import (
        SQLiteVSS,
    )
    from langchain_community.vectorstores.starrocks import (
        StarRocks,
    )
    from langchain_community.vectorstores.supabase import (
        SupabaseVectorStore,
    )
    from langchain_community.vectorstores.surrealdb import (
        SurrealDBStore,
    )
    from langchain_community.vectorstores.tair import (
        Tair,
    )
    from langchain_community.vectorstores.tencentvectordb import (
        TencentVectorDB,
    )
    from langchain_community.vectorstores.thirdai_neuraldb import (
        NeuralDBClientVectorStore,
        NeuralDBVectorStore,
    )
    from langchain_community.vectorstores.tidb_vector import (
        TiDBVectorStore,
    )
    from langchain_community.vectorstores.tigris import (
        Tigris,
    )
    from langchain_community.vectorstores.tiledb import (
        TileDB,
    )
    from langchain_community.vectorstores.timescalevector import (
        TimescaleVector,
    )
    from langchain_community.vectorstores.typesense import (
        Typesense,
    )
    from langchain_community.vectorstores.upstash import (
        UpstashVectorStore,
    )
    from langchain_community.vectorstores.usearch import (
        USearch,
    )
    from langchain_community.vectorstores.vald import (
        Vald,
    )
    from langchain_community.vectorstores.vdms import (
        VDMS,
    )
    from langchain_community.vectorstores.vearch import (
        Vearch,
    )
    from langchain_community.vectorstores.vectara import (
        Vectara,
    )
    from langchain_community.vectorstores.vespa import (
        VespaStore,
    )
    from langchain_community.vectorstores.vlite import (
        VLite,
    )
    from langchain_community.vectorstores.weaviate import (
        Weaviate,
    )
    from langchain_community.vectorstores.yellowbrick import (
        Yellowbrick,
    )
    from langchain_community.vectorstores.zep import (
        ZepVectorStore,
    )
    from langchain_community.vectorstores.zep_cloud import (
        ZepCloudVectorStore,
    )
    from langchain_community.vectorstores.zilliz import (
        Zilliz,
    )

__all__ = [
    "Aerospike",
    "AlibabaCloudOpenSearch",
    "AlibabaCloudOpenSearchSettings",
    "AnalyticDB",
    "Annoy",
    "ApacheDoris",
    "ApertureDB",
    "AstraDB",
    "AtlasDB",
    "AwaDB",
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBVectorSearch",
    "AzureSearch",
    "BESVectorStore",
    "Bagel",
    "BaiduVectorDB",
    "BigQueryVectorSearch",
    "Cassandra",
    "Chroma",
    "Clarifai",
    "Clickhouse",
    "ClickhouseSettings",
    "CouchbaseVectorStore",
    "DashVector",
    "DatabricksVectorSearch",
    "DeepLake",
    "Dingo",
    "DistanceStrategy",
    "DocArrayHnswSearch",
    "DocArrayInMemorySearch",
    "DocumentDBVectorSearch",
    "DuckDB",
    "EcloudESVectorStore",
    "ElasticKnnSearch",
    "ElasticVectorSearch",
    "ElasticsearchStore",
    "Epsilla",
    "FAISS",
    "HanaDB",
    "Hologres",
    "InMemoryVectorStore",
    "InfinispanVS",
    "KDBAI",
    "Kinetica",
    "KineticaSettings",
    "LLMRails",
    "LanceDB",
    "Lantern",
    "ManticoreSearch",
    "ManticoreSearchSettings",
    "Marqo",
    "MatchingEngine",
    "Meilisearch",
    "Milvus",
    "MomentoVectorIndex",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "Neo4jVector",
    "NeuralDBClientVectorStore",
    "NeuralDBVectorStore",
    "OceanBase",
    "OracleVS",
    "OpenSearchVectorSearch",
    "PGEmbedding",
    "PGVector",
    "PathwayVectorClient",
    "Pinecone",
    "Qdrant",
    "Redis",
    "Relyt",
    "Rockset",
    "SKLearnVectorStore",
    "SQLiteVec",
    "SQLiteVSS",
    "ScaNN",
    "SemaDB",
    "SingleStoreDB",
    "StarRocks",
    "SupabaseVectorStore",
    "SurrealDBStore",
    "Tair",
    "TencentVectorDB",
    "TiDBVectorStore",
    "Tigris",
    "TileDB",
    "TimescaleVector",
    "Typesense",
    "UpstashVectorStore",
    "USearch",
    "VDMS",
    "Vald",
    "Vearch",
    "Vectara",
    "VectorStore",
    "VespaStore",
    "VLite",
    "Weaviate",
    "Yellowbrick",
    "ZepVectorStore",
    "ZepCloudVectorStore",
    "Zilliz",
]

_module_lookup = {
    "Aerospike": "langchain_community.vectorstores.aerospike",
    "AlibabaCloudOpenSearch": "langchain_community.vectorstores.alibabacloud_opensearch",  # noqa: E501
    "AlibabaCloudOpenSearchSettings": "langchain_community.vectorstores.alibabacloud_opensearch",  # noqa: E501
    "AnalyticDB": "langchain_community.vectorstores.analyticdb",
    "Annoy": "langchain_community.vectorstores.annoy",
    "ApacheDoris": "langchain_community.vectorstores.apache_doris",
    "ApertureDB": "langchain_community.vectorstores.aperturedb",
    "AstraDB": "langchain_community.vectorstores.astradb",
    "AtlasDB": "langchain_community.vectorstores.atlas",
    "AwaDB": "langchain_community.vectorstores.awadb",
    "AzureCosmosDBNoSqlVectorSearch": "langchain_community.vectorstores.azure_cosmos_db_no_sql",  # noqa: E501
    "AzureCosmosDBVectorSearch": "langchain_community.vectorstores.azure_cosmos_db",  # noqa: E501
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
    "DuckDB": "langchain_community.vectorstores.duckdb",
    "EcloudESVectorStore": "langchain_community.vectorstores.ecloud_vector_search",
    "ElasticKnnSearch": "langchain_community.vectorstores.elastic_vector_search",
    "ElasticVectorSearch": "langchain_community.vectorstores.elastic_vector_search",
    "ElasticsearchStore": "langchain_community.vectorstores.elasticsearch",
    "Epsilla": "langchain_community.vectorstores.epsilla",
    "FAISS": "langchain_community.vectorstores.faiss",
    "HanaDB": "langchain_community.vectorstores.hanavector",
    "Hologres": "langchain_community.vectorstores.hologres",
    "InfinispanVS": "langchain_community.vectorstores.infinispanvs",
    "InMemoryVectorStore": "langchain_community.vectorstores.inmemory",
    "KDBAI": "langchain_community.vectorstores.kdbai",
    "Kinetica": "langchain_community.vectorstores.kinetica",
    "KineticaSettings": "langchain_community.vectorstores.kinetica",
    "LLMRails": "langchain_community.vectorstores.llm_rails",
    "LanceDB": "langchain_community.vectorstores.lancedb",
    "Lantern": "langchain_community.vectorstores.lantern",
    "ManticoreSearch": "langchain_community.vectorstores.manticore_search",
    "ManticoreSearchSettings": "langchain_community.vectorstores.manticore_search",
    "Marqo": "langchain_community.vectorstores.marqo",
    "MatchingEngine": "langchain_community.vectorstores.matching_engine",
    "Meilisearch": "langchain_community.vectorstores.meilisearch",
    "Milvus": "langchain_community.vectorstores.milvus",
    "MomentoVectorIndex": "langchain_community.vectorstores.momento_vector_index",
    "MongoDBAtlasVectorSearch": "langchain_community.vectorstores.mongodb_atlas",
    "MyScale": "langchain_community.vectorstores.myscale",
    "MyScaleSettings": "langchain_community.vectorstores.myscale",
    "Neo4jVector": "langchain_community.vectorstores.neo4j_vector",
    "NeuralDBClientVectorStore": "langchain_community.vectorstores.thirdai_neuraldb",
    "NeuralDBVectorStore": "langchain_community.vectorstores.thirdai_neuraldb",
    "OceanBase": "langchain_community.vectorstores.oceanbase",
    "OpenSearchVectorSearch": "langchain_community.vectorstores.opensearch_vector_search",  # noqa: E501
    "OracleVS": "langchain_community.vectorstores.oraclevs",
    "PathwayVectorClient": "langchain_community.vectorstores.pathway",
    "PGEmbedding": "langchain_community.vectorstores.pgembedding",
    "PGVector": "langchain_community.vectorstores.pgvector",
    "Pinecone": "langchain_community.vectorstores.pinecone",
    "Qdrant": "langchain_community.vectorstores.qdrant",
    "Redis": "langchain_community.vectorstores.redis",
    "Relyt": "langchain_community.vectorstores.relyt",
    "Rockset": "langchain_community.vectorstores.rocksetdb",
    "SKLearnVectorStore": "langchain_community.vectorstores.sklearn",
    "SQLiteVec": "langchain_community.vectorstores.sqlitevec",
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
    "UpstashVectorStore": "langchain_community.vectorstores.upstash",
    "USearch": "langchain_community.vectorstores.usearch",
    "Vald": "langchain_community.vectorstores.vald",
    "VDMS": "langchain_community.vectorstores.vdms",
    "Vearch": "langchain_community.vectorstores.vearch",
    "Vectara": "langchain_community.vectorstores.vectara",
    "VectorStore": "langchain_core.vectorstores",
    "VespaStore": "langchain_community.vectorstores.vespa",
    "VLite": "langchain_community.vectorstores.vlite",
    "Weaviate": "langchain_community.vectorstores.weaviate",
    "Yellowbrick": "langchain_community.vectorstores.yellowbrick",
    "ZepVectorStore": "langchain_community.vectorstores.zep",
    "ZepCloudVectorStore": "langchain_community.vectorstores.zep_cloud",
    "Zilliz": "langchain_community.vectorstores.zilliz",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
