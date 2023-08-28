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
from langchain_xfyun.vectorstores.alibabacloud_opensearch import (
    AlibabaCloudOpenSearch,
    AlibabaCloudOpenSearchSettings,
)
from langchain_xfyun.vectorstores.analyticdb import AnalyticDB
from langchain_xfyun.vectorstores.annoy import Annoy
from langchain_xfyun.vectorstores.atlas import AtlasDB
from langchain_xfyun.vectorstores.awadb import AwaDB
from langchain_xfyun.vectorstores.azuresearch import AzureSearch
from langchain_xfyun.vectorstores.bageldb import Bagel
from langchain_xfyun.vectorstores.base import VectorStore
from langchain_xfyun.vectorstores.cassandra import Cassandra
from langchain_xfyun.vectorstores.chroma import Chroma
from langchain_xfyun.vectorstores.clarifai import Clarifai
from langchain_xfyun.vectorstores.clickhouse import Clickhouse, ClickhouseSettings
from langchain_xfyun.vectorstores.dashvector import DashVector
from langchain_xfyun.vectorstores.deeplake import DeepLake
from langchain_xfyun.vectorstores.dingo import Dingo
from langchain_xfyun.vectorstores.docarray import DocArrayHnswSearch, DocArrayInMemorySearch
from langchain_xfyun.vectorstores.elastic_vector_search import (
    ElasticKnnSearch,
    ElasticVectorSearch,
)
from langchain_xfyun.vectorstores.elasticsearch import ElasticsearchStore
from langchain_xfyun.vectorstores.epsilla import Epsilla
from langchain_xfyun.vectorstores.faiss import FAISS
from langchain_xfyun.vectorstores.hologres import Hologres
from langchain_xfyun.vectorstores.lancedb import LanceDB
from langchain_xfyun.vectorstores.marqo import Marqo
from langchain_xfyun.vectorstores.matching_engine import MatchingEngine
from langchain_xfyun.vectorstores.meilisearch import Meilisearch
from langchain_xfyun.vectorstores.milvus import Milvus
from langchain_xfyun.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_xfyun.vectorstores.myscale import MyScale, MyScaleSettings
from langchain_xfyun.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_xfyun.vectorstores.pgembedding import PGEmbedding
from langchain_xfyun.vectorstores.pgvector import PGVector
from langchain_xfyun.vectorstores.pinecone import Pinecone
from langchain_xfyun.vectorstores.qdrant import Qdrant
from langchain_xfyun.vectorstores.redis import Redis
from langchain_xfyun.vectorstores.rocksetdb import Rockset
from langchain_xfyun.vectorstores.scann import ScaNN
from langchain_xfyun.vectorstores.singlestoredb import SingleStoreDB
from langchain_xfyun.vectorstores.sklearn import SKLearnVectorStore
from langchain_xfyun.vectorstores.starrocks import StarRocks
from langchain_xfyun.vectorstores.supabase import SupabaseVectorStore
from langchain_xfyun.vectorstores.tair import Tair
from langchain_xfyun.vectorstores.tigris import Tigris
from langchain_xfyun.vectorstores.typesense import Typesense
from langchain_xfyun.vectorstores.usearch import USearch
from langchain_xfyun.vectorstores.vectara import Vectara
from langchain_xfyun.vectorstores.weaviate import Weaviate
from langchain_xfyun.vectorstores.zep import ZepVectorStore
from langchain_xfyun.vectorstores.zilliz import Zilliz

__all__ = [
    "AlibabaCloudOpenSearch",
    "AlibabaCloudOpenSearchSettings",
    "AnalyticDB",
    "Annoy",
    "Annoy",
    "AtlasDB",
    "AtlasDB",
    "AwaDB",
    "AzureSearch",
    "Bagel",
    "Cassandra",
    "Chroma",
    "Chroma",
    "Clarifai",
    "Clickhouse",
    "ClickhouseSettings",
    "DashVector",
    "DeepLake",
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
    "Marqo",
    "MatchingEngine",
    "Meilisearch",
    "Milvus",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "OpenSearchVectorSearch",
    "OpenSearchVectorSearch",
    "PGEmbedding",
    "PGVector",
    "Pinecone",
    "Qdrant",
    "Redis",
    "Rockset",
    "SKLearnVectorStore",
    "ScaNN",
    "SingleStoreDB",
    "SingleStoreDB",
    "StarRocks",
    "SupabaseVectorStore",
    "Tair",
    "Tigris",
    "Typesense",
    "USearch",
    "Vectara",
    "VectorStore",
    "Weaviate",
    "ZepVectorStore",
    "Zilliz",
    "Zilliz",
]
