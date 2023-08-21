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
from langchain.vectorstores.alibabacloud_opensearch import (
    AlibabaCloudOpenSearch,
    AlibabaCloudOpenSearchSettings,
)
from langchain.vectorstores.analyticdb import AnalyticDB
from langchain.vectorstores.annoy import Annoy
from langchain.vectorstores.atlas import AtlasDB
from langchain.vectorstores.awadb import AwaDB
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.vectorstores.bageldb import Bagel
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.cassandra import Cassandra
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.clarifai import Clarifai
from langchain.vectorstores.clickhouse import Clickhouse, ClickhouseSettings
from langchain.vectorstores.dashvector import DashVector
from langchain.vectorstores.deeplake import DeepLake
from langchain.vectorstores.dingo import Dingo
from langchain.vectorstores.docarray import DocArrayHnswSearch, DocArrayInMemorySearch
from langchain.vectorstores.elastic_vector_search import (
    ElasticKnnSearch,
    ElasticVectorSearch,
)
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.hologres import Hologres
from langchain.vectorstores.lancedb import LanceDB
from langchain.vectorstores.marqo import Marqo
from langchain.vectorstores.matching_engine import MatchingEngine
from langchain.vectorstores.meilisearch import Meilisearch
from langchain.vectorstores.milvus import Milvus
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.vectorstores.myscale import MyScale, MyScaleSettings
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain.vectorstores.pgembedding import PGEmbedding
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pinecone import Pinecone
from langchain.vectorstores.qdrant import Qdrant
from langchain.vectorstores.redis import Redis
from langchain.vectorstores.rocksetdb import Rockset
from langchain.vectorstores.scann import ScaNN
from langchain.vectorstores.singlestoredb import SingleStoreDB
from langchain.vectorstores.sklearn import SKLearnVectorStore
from langchain.vectorstores.starrocks import StarRocks
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.vectorstores.tair import Tair
from langchain.vectorstores.tigris import Tigris
from langchain.vectorstores.typesense import Typesense
from langchain.vectorstores.usearch import USearch
from langchain.vectorstores.vectara import Vectara
from langchain.vectorstores.weaviate import Weaviate
from langchain.vectorstores.zep import ZepVectorStore
from langchain.vectorstores.zilliz import Zilliz

__all__ = [
    "AlibabaCloudOpenSearch",
    "AlibabaCloudOpenSearchSettings",
    "AnalyticDB",
    "Annoy",
    "AtlasDB",
    "AwaDB",
    "AzureSearch",
    "Bagel",
    "Cassandra",
    "Chroma",
    "Clickhouse",
    "ClickhouseSettings",
    "DashVector",
    "DeepLake",
    "Dingo",
    "DocArrayHnswSearch",
    "DocArrayInMemorySearch",
    "ElasticVectorSearch",
    "ElasticKnnSearch",
    "ElasticsearchStore",
    "FAISS",
    "PGEmbedding",
    "Hologres",
    "LanceDB",
    "MatchingEngine",
    "Marqo",
    "Meilisearch",
    "Milvus",
    "Zilliz",
    "SingleStoreDB",
    "Chroma",
    "Clarifai",
    "OpenSearchVectorSearch",
    "AtlasDB",
    "DeepLake",
    "Annoy",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "OpenSearchVectorSearch",
    "Pinecone",
    "Qdrant",
    "Redis",
    "Rockset",
    "ScaNN",
    "SKLearnVectorStore",
    "SingleStoreDB",
    "StarRocks",
    "SupabaseVectorStore",
    "Tair",
    "Tigris",
    "Typesense",
    "Vectara",
    "VectorStore",
    "Weaviate",
    "ZepVectorStore",
    "Zilliz",
    "PGVector",
    "USearch",
]
