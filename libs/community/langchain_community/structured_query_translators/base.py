"""Retriever that generates and executes structured queries over its own data source."""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.structured_query.ir import StructuredQuery, Visitor
from langchain_core.vectorstores import VectorStore

from langchain_community.structured_query_translators.astradb import AstraDBTranslator
from langchain_community.structured_query_translators.chroma import ChromaTranslator
from langchain_community.structured_query_translators.dashvector import (
    DashvectorTranslator,
)
from langchain_community.structured_query_translators.deeplake import DeepLakeTranslator
from langchain_community.structured_query_translators.dingo import DingoDBTranslator
from langchain_community.structured_query_translators.elasticsearch import (
    ElasticsearchTranslator,
)
from langchain_community.structured_query_translators.milvus import MilvusTranslator
from langchain_community.structured_query_translators.mongodb_atlas import (
    MongoDBAtlasTranslator,
)
from langchain_community.structured_query_translators.myscale import MyScaleTranslator
from langchain_community.structured_query_translators.opensearch import (
    OpenSearchTranslator,
)
from langchain_community.structured_query_translators.pgvector import PGVectorTranslator
from langchain_community.structured_query_translators.pinecone import PineconeTranslator
from langchain_community.structured_query_translators.qdrant import QdrantTranslator
from langchain_community.structured_query_translators.redis import RedisTranslator
from langchain_community.structured_query_translators.supabase import (
    SupabaseVectorTranslator,
)
from langchain_community.structured_query_translators.timescalevector import (
    TimescaleVectorTranslator,
)
from langchain_community.structured_query_translators.vectara import VectaraTranslator
from langchain_community.structured_query_translators.weaviate import WeaviateTranslator

logger = logging.getLogger(__name__)
QUERY_CONSTRUCTOR_RUN_NAME = "query_constructor"


def _get_builtin_translator(vectorstore: VectorStore) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
        "AstraDB": AstraDBTranslator,
        "PGVector": PGVectorTranslator,
        "Pinecone": PineconeTranslator,
        "Chroma": ChromaTranslator,
        "DashVector": DashvectorTranslator,
        "Dingo": DingoDBTranslator,
        "Weaviate": WeaviateTranslator,
        "Vectara": VectaraTranslator,
        "Qdrant": QdrantTranslator,
        "MyScale": MyScaleTranslator,
        "DeepLake": DeepLakeTranslator,
        "ElasticsearchStore": ElasticsearchTranslator,
        "Milvus": MilvusTranslator,
        "SupabaseVectorStore": SupabaseVectorTranslator,
        "TimescaleVector": TimescaleVectorTranslator,
        "OpenSearchVectorSearch": OpenSearchTranslator,
        "MongoDBAtlasVectorSearch": MongoDBAtlasTranslator,
    }

    # TODO: This is a hack so we don't have to import langchain-community, fix.
    if vectorstore.__class__.__name__.startswith("langchain_community.vectorstores"):
        if vectorstore.__class__.__name__ == "Qdrant":
            return QdrantTranslator(metadata_key=vectorstore.metadata_payload_key)
        elif vectorstore.__class__.__name__ == "MyScale":
            return MyScaleTranslator(metadata_key=vectorstore.metadata_column)
        elif vectorstore.__class__.__name__ == "Redis":
            return RedisTranslator.from_vectorstore(vectorstore)
        elif vectorstore.__class__.__name__ in BUILTIN_TRANSLATORS:
            return BUILTIN_TRANSLATORS[vectorstore.__class__]()
    else:
        try:
            from langchain_astradb.vectorstores import AstraDBVectorStore

            if isinstance(vectorstore, AstraDBVectorStore):
                return AstraDBTranslator()
        except ImportError:
            pass

    raise ValueError(
        f"Self query retriever with Vector Store type {vectorstore.__class__}"
        f" not supported."
    )