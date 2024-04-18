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
        VectorStore,  # noqa: F401
    )

    from langchain_community.vectorstores.alibabacloud_opensearch import (
        AlibabaCloudOpenSearch,  # noqa: F401
        AlibabaCloudOpenSearchSettings,  # noqa: F401
    )
    from langchain_community.vectorstores.analyticdb import (
        AnalyticDB,  # noqa: F401
    )
    from langchain_community.vectorstores.annoy import (
        Annoy,  # noqa: F401
    )
    from langchain_community.vectorstores.apache_doris import (
        ApacheDoris,  # noqa: F401
    )
    from langchain_community.vectorstores.astradb import (
        AstraDB,  # noqa: F401
    )
    from langchain_community.vectorstores.atlas import (
        AtlasDB,  # noqa: F401
    )
    from langchain_community.vectorstores.awadb import (
        AwaDB,  # noqa: F401
    )
    from langchain_community.vectorstores.azure_cosmos_db import (
        AzureCosmosDBVectorSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.azuresearch import (
        AzureSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.bageldb import (
        Bagel,  # noqa: F401
    )
    from langchain_community.vectorstores.baiducloud_vector_search import (
        BESVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.baiduvectordb import (
        BaiduVectorDB,  # noqa: F401
    )
    from langchain_community.vectorstores.bigquery_vector_search import (
        BigQueryVectorSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.cassandra import (
        Cassandra,  # noqa: F401
    )
    from langchain_community.vectorstores.chroma import (
        Chroma,  # noqa: F401
    )
    from langchain_community.vectorstores.clarifai import (
        Clarifai,  # noqa: F401
    )
    from langchain_community.vectorstores.clickhouse import (
        Clickhouse,  # noqa: F401
        ClickhouseSettings,  # noqa: F401
    )
    from langchain_community.vectorstores.couchbase import (
        CouchbaseVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.dashvector import (
        DashVector,  # noqa: F401
    )
    from langchain_community.vectorstores.databricks_vector_search import (
        DatabricksVectorSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.deeplake import (
        DeepLake,  # noqa: F401
    )
    from langchain_community.vectorstores.dingo import (
        Dingo,  # noqa: F401
    )
    from langchain_community.vectorstores.docarray import (
        DocArrayHnswSearch,  # noqa: F401
        DocArrayInMemorySearch,  # noqa: F401
    )
    from langchain_community.vectorstores.documentdb import (
        DocumentDBVectorSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.duckdb import (
        DuckDB,  # noqa: F401
    )
    from langchain_community.vectorstores.ecloud_vector_search import (
        EcloudESVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.elastic_vector_search import (
        ElasticKnnSearch,  # noqa: F401
        ElasticVectorSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.elasticsearch import (
        ElasticsearchStore,  # noqa: F401
    )
    from langchain_community.vectorstores.epsilla import (
        Epsilla,  # noqa: F401
    )
    from langchain_community.vectorstores.faiss import (
        FAISS,  # noqa: F401
    )
    from langchain_community.vectorstores.hanavector import (
        HanaDB,  # noqa: F401
    )
    from langchain_community.vectorstores.hologres import (
        Hologres,  # noqa: F401
    )
    from langchain_community.vectorstores.infinispanvs import (
        InfinispanVS,  # noqa: F401
    )
    from langchain_community.vectorstores.inmemory import (
        InMemoryVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.kdbai import (
        KDBAI,  # noqa: F401
    )
    from langchain_community.vectorstores.kinetica import (
        DistanceStrategy,  # noqa: F401
        Kinetica,  # noqa: F401
        KineticaSettings,  # noqa: F401
    )
    from langchain_community.vectorstores.lancedb import (
        LanceDB,  # noqa: F401
    )
    from langchain_community.vectorstores.lantern import (
        Lantern,  # noqa: F401
    )
    from langchain_community.vectorstores.llm_rails import (
        LLMRails,  # noqa: F401
    )
    from langchain_community.vectorstores.marqo import (
        Marqo,  # noqa: F401
    )
    from langchain_community.vectorstores.matching_engine import (
        MatchingEngine,  # noqa: F401
    )
    from langchain_community.vectorstores.meilisearch import (
        Meilisearch,  # noqa: F401
    )
    from langchain_community.vectorstores.milvus import (
        Milvus,  # noqa: F401
    )
    from langchain_community.vectorstores.momento_vector_index import (
        MomentoVectorIndex,  # noqa: F401
    )
    from langchain_community.vectorstores.mongodb_atlas import (
        MongoDBAtlasVectorSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.myscale import (
        MyScale,  # noqa: F401
        MyScaleSettings,  # noqa: F401
    )
    from langchain_community.vectorstores.neo4j_vector import (
        Neo4jVector,  # noqa: F401
    )
    from langchain_community.vectorstores.opensearch_vector_search import (
        OpenSearchVectorSearch,  # noqa: F401
    )
    from langchain_community.vectorstores.pathway import (
        PathwayVectorClient,  # noqa: F401
    )
    from langchain_community.vectorstores.pgembedding import (
        PGEmbedding,  # noqa: F401
    )
    from langchain_community.vectorstores.pgvector import (
        PGVector,  # noqa: F401
    )
    from langchain_community.vectorstores.pinecone import (
        Pinecone,  # noqa: F401
    )
    from langchain_community.vectorstores.qdrant import (
        Qdrant,  # noqa: F401
    )
    from langchain_community.vectorstores.redis import (
        Redis,  # noqa: F401
    )
    from langchain_community.vectorstores.rocksetdb import (
        Rockset,  # noqa: F401
    )
    from langchain_community.vectorstores.scann import (
        ScaNN,  # noqa: F401
    )
    from langchain_community.vectorstores.semadb import (
        SemaDB,  # noqa: F401
    )
    from langchain_community.vectorstores.singlestoredb import (
        SingleStoreDB,  # noqa: F401
    )
    from langchain_community.vectorstores.sklearn import (
        SKLearnVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.sqlitevss import (
        SQLiteVSS,  # noqa: F401
    )
    from langchain_community.vectorstores.starrocks import (
        StarRocks,  # noqa: F401
    )
    from langchain_community.vectorstores.supabase import (
        SupabaseVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.surrealdb import (
        SurrealDBStore,  # noqa: F401
    )
    from langchain_community.vectorstores.tair import (
        Tair,  # noqa: F401
    )
    from langchain_community.vectorstores.tencentvectordb import (
        TencentVectorDB,  # noqa: F401
    )
    from langchain_community.vectorstores.thirdai_neuraldb import (
        NeuralDBVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.tidb_vector import (
        TiDBVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.tigris import (
        Tigris,  # noqa: F401
    )
    from langchain_community.vectorstores.tiledb import (
        TileDB,  # noqa: F401
    )
    from langchain_community.vectorstores.timescalevector import (
        TimescaleVector,  # noqa: F401
    )
    from langchain_community.vectorstores.typesense import (
        Typesense,  # noqa: F401
    )
    from langchain_community.vectorstores.usearch import (
        USearch,  # noqa: F401
    )
    from langchain_community.vectorstores.vald import (
        Vald,  # noqa: F401
    )
    from langchain_community.vectorstores.vdms import (
        VDMS,  # noqa: F401
    )
    from langchain_community.vectorstores.vearch import (
        Vearch,  # noqa: F401
    )
    from langchain_community.vectorstores.vectara import (
        Vectara,  # noqa: F401
    )
    from langchain_community.vectorstores.vespa import (
        VespaStore,  # noqa: F401
    )
    from langchain_community.vectorstores.vlite import (
        VLite,  # noqa: F401
    )
    from langchain_community.vectorstores.weaviate import (
        Weaviate,  # noqa: F401
    )
    from langchain_community.vectorstores.yellowbrick import (
        Yellowbrick,  # noqa: F401
    )
    from langchain_community.vectorstores.zep import (
        ZepVectorStore,  # noqa: F401
    )
    from langchain_community.vectorstores.zilliz import (
        Zilliz,  # noqa: F401
    )

__all__ = [
    "AlibabaCloudOpenSearch",
    "AlibabaCloudOpenSearchSettings",
    "AnalyticDB",
    "Annoy",
    "ApacheDoris",
    "AstraDB",
    "AtlasDB",
    "AwaDB",
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
    "Marqo",
    "MatchingEngine",
    "Meilisearch",
    "Milvus",
    "MomentoVectorIndex",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "Neo4jVector",
    "NeuralDBVectorStore",
    "OpenSearchVectorSearch",
    "PGEmbedding",
    "PGVector",
    "PathwayVectorClient",
    "Pinecone",
    "Qdrant",
    "Redis",
    "Rockset",
    "SKLearnVectorStore",
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
    "Zilliz",
]

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
    "PathwayVectorClient": "langchain_community.vectorstores.pathway",
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
    "VDMS": "langchain_community.vectorstores.vdms",
    "Vearch": "langchain_community.vectorstores.vearch",
    "Vectara": "langchain_community.vectorstores.vectara",
    "VectorStore": "langchain_core.vectorstores",
    "VespaStore": "langchain_community.vectorstores.vespa",
    "VLite": "langchain_community.vectorstores.vlite",
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
