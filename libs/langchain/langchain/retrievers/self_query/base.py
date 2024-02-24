"""Retriever that generates and executes structured queries over its own data source."""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from langchain_community.vectorstores import (
    AstraDB,
    Chroma,
    DashVector,
    DeepLake,
    ElasticsearchStore,
    Milvus,
    MongoDBAtlasVectorSearch,
    MyScale,
    OpenSearchVectorSearch,
    PGVector,
    Pinecone,
    Qdrant,
    Redis,
    SupabaseVectorStore,
    TimescaleVector,
    Vectara,
    Weaviate,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.chains.query_constructor.ir import StructuredQuery, Visitor
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.astradb import AstraDBTranslator
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.retrievers.self_query.dashvector import DashvectorTranslator
from langchain.retrievers.self_query.deeplake import DeepLakeTranslator
from langchain.retrievers.self_query.elasticsearch import ElasticsearchTranslator
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain.retrievers.self_query.mongodb_atlas import MongoDBAtlasTranslator
from langchain.retrievers.self_query.myscale import MyScaleTranslator
from langchain.retrievers.self_query.opensearch import OpenSearchTranslator
from langchain.retrievers.self_query.pgvector import PGVectorTranslator
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain.retrievers.self_query.qdrant import QdrantTranslator
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.retrievers.self_query.supabase import SupabaseVectorTranslator
from langchain.retrievers.self_query.timescalevector import TimescaleVectorTranslator
from langchain.retrievers.self_query.vectara import VectaraTranslator
from langchain.retrievers.self_query.weaviate import WeaviateTranslator

logger = logging.getLogger(__name__)


def _get_builtin_translator(vectorstore: VectorStore) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
        AstraDB: AstraDBTranslator,
        PGVector: PGVectorTranslator,
        Pinecone: PineconeTranslator,
        Chroma: ChromaTranslator,
        DashVector: DashvectorTranslator,
        Weaviate: WeaviateTranslator,
        Vectara: VectaraTranslator,
        Qdrant: QdrantTranslator,
        MyScale: MyScaleTranslator,
        DeepLake: DeepLakeTranslator,
        ElasticsearchStore: ElasticsearchTranslator,
        Milvus: MilvusTranslator,
        SupabaseVectorStore: SupabaseVectorTranslator,
        TimescaleVector: TimescaleVectorTranslator,
        OpenSearchVectorSearch: OpenSearchTranslator,
        MongoDBAtlasVectorSearch: MongoDBAtlasTranslator,
    }

    if isinstance(vectorstore, Qdrant):
        return QdrantTranslator(metadata_key=vectorstore.metadata_payload_key)
    elif isinstance(vectorstore, MyScale):
        return MyScaleTranslator(metadata_key=vectorstore.metadata_column)
    elif isinstance(vectorstore, Redis):
        return RedisTranslator.from_vectorstore(vectorstore)
    elif vectorstore.__class__ in BUILTIN_TRANSLATORS:
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


class SelfQueryRetriever(BaseRetriever):
    """Retriever that uses a vector store and an LLM to generate
    the vector store queries."""

    vectorstore: VectorStore
    """The underlying vector store from which documents will be retrieved."""
    query_constructor: Runnable[dict, StructuredQuery] = Field(alias="llm_chain")
    """The query constructor chain for generating the vector store queries.
    
    llm_chain is legacy name kept for backwards compatibility."""
    search_type: str = "similarity"
    """The search type to perform on the vector store."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass in to the vector store search."""
    structured_query_translator: Visitor
    """Translator for turning internal query language into vectorstore search params."""
    verbose: bool = False

    use_original_query: bool = False
    """Use original query instead of the revised new query from LLM"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def validate_translator(cls, values: Dict) -> Dict:
        """Validate translator."""
        if "structured_query_translator" not in values:
            values["structured_query_translator"] = _get_builtin_translator(
                values["vectorstore"]
            )
        return values

    @property
    def llm_chain(self) -> Runnable:
        """llm_chain is legacy name kept for backwards compatibility."""
        return self.query_constructor

    def _prepare_query(
        self, query: str, structured_query: StructuredQuery
    ) -> Tuple[str, Dict[str, Any]]:
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query
        )
        if structured_query.limit is not None:
            new_kwargs["k"] = structured_query.limit
        if self.use_original_query:
            new_query = query
        search_kwargs = {**self.search_kwargs, **new_kwargs}
        return new_query, search_kwargs

    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs = self.vectorstore.search(query, self.search_type, **search_kwargs)
        return docs

    async def _aget_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs = await self.vectorstore.asearch(query, self.search_type, **search_kwargs)
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        structured_query = self.query_constructor.invoke(
            {"query": query}, config={"callbacks": run_manager.get_child()}
        )
        if self.verbose:
            logger.info(f"Generated Query: {structured_query}")
        new_query, search_kwargs = self._prepare_query(query, structured_query)
        docs = self._get_docs_with_query(new_query, search_kwargs)
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        structured_query = await self.query_constructor.ainvoke(
            {"query": query}, config={"callbacks": run_manager.get_child()}
        )
        if self.verbose:
            logger.info(f"Generated Query: {structured_query}")
        new_query, search_kwargs = self._prepare_query(query, structured_query)
        docs = await self._aget_docs_with_query(new_query, search_kwargs)
        return docs

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        document_contents: str,
        metadata_field_info: Sequence[Union[AttributeInfo, dict]],
        structured_query_translator: Optional[Visitor] = None,
        chain_kwargs: Optional[Dict] = None,
        enable_limit: bool = False,
        use_original_query: bool = False,
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        if structured_query_translator is None:
            structured_query_translator = _get_builtin_translator(vectorstore)
        chain_kwargs = chain_kwargs or {}

        if (
            "allowed_comparators" not in chain_kwargs
            and structured_query_translator.allowed_comparators is not None
        ):
            chain_kwargs[
                "allowed_comparators"
            ] = structured_query_translator.allowed_comparators
        if (
            "allowed_operators" not in chain_kwargs
            and structured_query_translator.allowed_operators is not None
        ):
            chain_kwargs[
                "allowed_operators"
            ] = structured_query_translator.allowed_operators
        query_constructor = load_query_constructor_runnable(
            llm,
            document_contents,
            metadata_field_info,
            enable_limit=enable_limit,
            **chain_kwargs,
        )
        return cls(
            query_constructor=query_constructor,
            vectorstore=vectorstore,
            use_original_query=use_original_query,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )
