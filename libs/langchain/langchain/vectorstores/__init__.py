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
import warnings
from typing import Any

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.vectorstores import VectorStore

from langchain.utils.interactive_env import is_interactive_env
from langchain._api import create_importer

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from langchain_community.vectorstores import AlibabaCloudOpenSearch
    from langchain_community.vectorstores import AlibabaCloudOpenSearchSettings
    from langchain_community.vectorstores import AnalyticDB
    from langchain_community.vectorstores import Annoy
    from langchain_community.vectorstores import AtlasDB
    from langchain_community.vectorstores import AwaDB
    from langchain_community.vectorstores import AzureSearch
    from langchain_community.vectorstores import Bagel
    from langchain_community.vectorstores import Cassandra
    from langchain_community.vectorstores import AstraDB
    from langchain_community.vectorstores import Chroma
    from langchain_community.vectorstores import Clarifai
    from langchain_community.vectorstores import Clickhouse
    from langchain_community.vectorstores import ClickhouseSettings
    from langchain_community.vectorstores import DashVector
    from langchain_community.vectorstores import DatabricksVectorSearch
    from langchain_community.vectorstores import DeepLake
    from langchain_community.vectorstores import Dingo
    from langchain_community.vectorstores import DocArrayHnswSearch
    from langchain_community.vectorstores import DocArrayInMemorySearch
    from langchain_community.vectorstores import ElasticKnnSearch
    from langchain_community.vectorstores import ElasticVectorSearch
    from langchain_community.vectorstores import ElasticsearchStore
    from langchain_community.vectorstores import Epsilla
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores import Hologres
    from langchain_community.vectorstores import LanceDB
    from langchain_community.vectorstores import LLMRails
    from langchain_community.vectorstores import Marqo
    from langchain_community.vectorstores import MatchingEngine
    from langchain_community.vectorstores import Meilisearch
    from langchain_community.vectorstores import Milvus
    from langchain_community.vectorstores import MomentoVectorIndex
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch
    from langchain_community.vectorstores import MyScale
    from langchain_community.vectorstores import MyScaleSettings
    from langchain_community.vectorstores import Neo4jVector
    from langchain_community.vectorstores import OpenSearchVectorSearch
    from langchain_community.vectorstores import PGEmbedding
    from langchain_community.vectorstores import PGVector
    from langchain_community.vectorstores import Pinecone
    from langchain_community.vectorstores import Qdrant
    from langchain_community.vectorstores import Redis
    from langchain_community.vectorstores import Rockset
    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_community.vectorstores import ScaNN
    from langchain_community.vectorstores import SemaDB
    from langchain_community.vectorstores import SingleStoreDB
    from langchain_community.vectorstores import SQLiteVSS
    from langchain_community.vectorstores import StarRocks
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_community.vectorstores import Tair
    from langchain_community.vectorstores import TileDB
    from langchain_community.vectorstores import Tigris
    from langchain_community.vectorstores import TimescaleVector
    from langchain_community.vectorstores import Typesense
    from langchain_community.vectorstores import USearch
    from langchain_community.vectorstores import Vald
    from langchain_community.vectorstores import Vearch
    from langchain_community.vectorstores import Vectara
    from langchain_community.vectorstores import VespaStore
    from langchain_community.vectorstores import Weaviate
    from langchain_community.vectorstores import Yellowbrick
    from langchain_community.vectorstores import ZepVectorStore
    from langchain_community.vectorstores import Zilliz
    from langchain_community.vectorstores import TencentVectorDB
    from langchain_community.vectorstores import AzureCosmosDBVectorSearch
            
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {"AlibabaCloudOpenSearch": "langchain_community.vectorstores", "AlibabaCloudOpenSearchSettings": "langchain_community.vectorstores", "AnalyticDB": "langchain_community.vectorstores", "Annoy": "langchain_community.vectorstores", "AtlasDB": "langchain_community.vectorstores", "AwaDB": "langchain_community.vectorstores", "AzureSearch": "langchain_community.vectorstores", "Bagel": "langchain_community.vectorstores", "Cassandra": "langchain_community.vectorstores", "AstraDB": "langchain_community.vectorstores", "Chroma": "langchain_community.vectorstores", "Clarifai": "langchain_community.vectorstores", "Clickhouse": "langchain_community.vectorstores", "ClickhouseSettings": "langchain_community.vectorstores", "DashVector": "langchain_community.vectorstores", "DatabricksVectorSearch": "langchain_community.vectorstores", "DeepLake": "langchain_community.vectorstores", "Dingo": "langchain_community.vectorstores", "DocArrayHnswSearch": "langchain_community.vectorstores", "DocArrayInMemorySearch": "langchain_community.vectorstores", "ElasticKnnSearch": "langchain_community.vectorstores", "ElasticVectorSearch": "langchain_community.vectorstores", "ElasticsearchStore": "langchain_community.vectorstores", "Epsilla": "langchain_community.vectorstores", "FAISS": "langchain_community.vectorstores", "Hologres": "langchain_community.vectorstores", "LanceDB": "langchain_community.vectorstores", "LLMRails": "langchain_community.vectorstores", "Marqo": "langchain_community.vectorstores", "MatchingEngine": "langchain_community.vectorstores", "Meilisearch": "langchain_community.vectorstores", "Milvus": "langchain_community.vectorstores", "MomentoVectorIndex": "langchain_community.vectorstores", "MongoDBAtlasVectorSearch": "langchain_community.vectorstores", "MyScale": "langchain_community.vectorstores", "MyScaleSettings": "langchain_community.vectorstores", "Neo4jVector": "langchain_community.vectorstores", "OpenSearchVectorSearch": "langchain_community.vectorstores", "PGEmbedding": "langchain_community.vectorstores", "PGVector": "langchain_community.vectorstores", "Pinecone": "langchain_community.vectorstores", "Qdrant": "langchain_community.vectorstores", "Redis": "langchain_community.vectorstores", "Rockset": "langchain_community.vectorstores", "SKLearnVectorStore": "langchain_community.vectorstores", "ScaNN": "langchain_community.vectorstores", "SemaDB": "langchain_community.vectorstores", "SingleStoreDB": "langchain_community.vectorstores", "SQLiteVSS": "langchain_community.vectorstores", "StarRocks": "langchain_community.vectorstores", "SupabaseVectorStore": "langchain_community.vectorstores", "Tair": "langchain_community.vectorstores", "TileDB": "langchain_community.vectorstores", "Tigris": "langchain_community.vectorstores", "TimescaleVector": "langchain_community.vectorstores", "Typesense": "langchain_community.vectorstores", "USearch": "langchain_community.vectorstores", "Vald": "langchain_community.vectorstores", "Vearch": "langchain_community.vectorstores", "Vectara": "langchain_community.vectorstores", "VespaStore": "langchain_community.vectorstores", "Weaviate": "langchain_community.vectorstores", "Yellowbrick": "langchain_community.vectorstores", "ZepVectorStore": "langchain_community.vectorstores", "Zilliz": "langchain_community.vectorstores", "TencentVectorDB": "langchain_community.vectorstores", "AzureCosmosDBVectorSearch": "langchain_community.vectorstores"}
        
_import_attribute=create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
__all__ = ["AlibabaCloudOpenSearch",
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
"VectorStore",
]
