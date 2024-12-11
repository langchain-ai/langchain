"""Retriever that generates and executes structured queries over its own data source."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.structured_query import StructuredQuery, Visitor
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field, model_validator

from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.chains.query_constructor.schema import AttributeInfo

logger = logging.getLogger(__name__)
QUERY_CONSTRUCTOR_RUN_NAME = "query_constructor"


def _get_builtin_translator(vectorstore: VectorStore) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    try:
        import langchain_community  # noqa: F401
    except ImportError:
        raise ImportError(
            "The langchain-community package must be installed to use this feature."
            " Please install it using `pip install langchain-community`."
        )

    from langchain_community.query_constructors.astradb import AstraDBTranslator
    from langchain_community.query_constructors.chroma import ChromaTranslator
    from langchain_community.query_constructors.dashvector import DashvectorTranslator
    from langchain_community.query_constructors.databricks_vector_search import (
        DatabricksVectorSearchTranslator,
    )
    from langchain_community.query_constructors.deeplake import DeepLakeTranslator
    from langchain_community.query_constructors.dingo import DingoDBTranslator
    from langchain_community.query_constructors.elasticsearch import (
        ElasticsearchTranslator,
    )
    from langchain_community.query_constructors.milvus import MilvusTranslator
    from langchain_community.query_constructors.mongodb_atlas import (
        MongoDBAtlasTranslator,
    )
    from langchain_community.query_constructors.myscale import MyScaleTranslator
    from langchain_community.query_constructors.neo4j import Neo4jTranslator
    from langchain_community.query_constructors.opensearch import OpenSearchTranslator
    from langchain_community.query_constructors.pgvector import PGVectorTranslator
    from langchain_community.query_constructors.pinecone import PineconeTranslator
    from langchain_community.query_constructors.qdrant import QdrantTranslator
    from langchain_community.query_constructors.redis import RedisTranslator
    from langchain_community.query_constructors.supabase import SupabaseVectorTranslator
    from langchain_community.query_constructors.tencentvectordb import (
        TencentVectorDBTranslator,
    )
    from langchain_community.query_constructors.timescalevector import (
        TimescaleVectorTranslator,
    )
    from langchain_community.query_constructors.vectara import VectaraTranslator
    from langchain_community.query_constructors.weaviate import WeaviateTranslator
    from langchain_community.vectorstores import (
        AstraDB,
        DashVector,
        DatabricksVectorSearch,
        DeepLake,
        Dingo,
        Milvus,
        MyScale,
        Neo4jVector,
        OpenSearchVectorSearch,
        PGVector,
        Qdrant,
        Redis,
        SupabaseVectorStore,
        TencentVectorDB,
        TimescaleVector,
        Vectara,
        Weaviate,
    )
    from langchain_community.vectorstores import (
        Chroma as CommunityChroma,
    )
    from langchain_community.vectorstores import (
        ElasticsearchStore as ElasticsearchStoreCommunity,
    )
    from langchain_community.vectorstores import (
        MongoDBAtlasVectorSearch as CommunityMongoDBAtlasVectorSearch,
    )
    from langchain_community.vectorstores import (
        Pinecone as CommunityPinecone,
    )

    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
        AstraDB: AstraDBTranslator,
        PGVector: PGVectorTranslator,
        CommunityPinecone: PineconeTranslator,
        CommunityChroma: ChromaTranslator,
        DashVector: DashvectorTranslator,
        Dingo: DingoDBTranslator,
        Weaviate: WeaviateTranslator,
        Vectara: VectaraTranslator,
        Qdrant: QdrantTranslator,
        MyScale: MyScaleTranslator,
        DeepLake: DeepLakeTranslator,
        ElasticsearchStoreCommunity: ElasticsearchTranslator,
        Milvus: MilvusTranslator,
        SupabaseVectorStore: SupabaseVectorTranslator,
        TimescaleVector: TimescaleVectorTranslator,
        OpenSearchVectorSearch: OpenSearchTranslator,
        CommunityMongoDBAtlasVectorSearch: MongoDBAtlasTranslator,
        Neo4jVector: Neo4jTranslator,
    }
    if isinstance(vectorstore, DatabricksVectorSearch):
        return DatabricksVectorSearchTranslator()
    elif isinstance(vectorstore, MyScale):
        return MyScaleTranslator(metadata_key=vectorstore.metadata_column)
    elif isinstance(vectorstore, Redis):
        return RedisTranslator.from_vectorstore(vectorstore)
    elif isinstance(vectorstore, TencentVectorDB):
        fields = [
            field.name for field in (vectorstore.meta_fields or []) if field.index
        ]
        return TencentVectorDBTranslator(fields)
    elif vectorstore.__class__ in BUILTIN_TRANSLATORS:
        return BUILTIN_TRANSLATORS[vectorstore.__class__]()
    else:
        try:
            from langchain_astradb.vectorstores import AstraDBVectorStore
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, AstraDBVectorStore):
                return AstraDBTranslator()

        try:
            from langchain_elasticsearch.vectorstores import ElasticsearchStore
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, ElasticsearchStore):
                return ElasticsearchTranslator()

        try:
            from langchain_pinecone import PineconeVectorStore
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, PineconeVectorStore):
                return PineconeTranslator()

        try:
            from langchain_mongodb import MongoDBAtlasVectorSearch
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, MongoDBAtlasVectorSearch):
                return MongoDBAtlasTranslator()

        try:
            from langchain_neo4j import Neo4jVector
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, Neo4jVector):
                return Neo4jTranslator()

        try:
            from langchain_chroma import Chroma
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, Chroma):
                return ChromaTranslator()

        try:
            from langchain_postgres import PGVector  # type: ignore[no-redef]
            from langchain_postgres import PGVectorTranslator as NewPGVectorTranslator
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, PGVector):
                return NewPGVectorTranslator()

        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, QdrantVectorStore):
                return QdrantTranslator(metadata_key=vectorstore.metadata_payload_key)

        try:
            # Added in langchain-community==0.2.11
            from langchain_community.query_constructors.hanavector import HanaTranslator
            from langchain_community.vectorstores import HanaDB
        except ImportError:
            pass
        else:
            if isinstance(vectorstore, HanaDB):
                return HanaTranslator()

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

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_translator(cls, values: Dict) -> Any:
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
            chain_kwargs["allowed_comparators"] = (
                structured_query_translator.allowed_comparators
            )
        if (
            "allowed_operators" not in chain_kwargs
            and structured_query_translator.allowed_operators is not None
        ):
            chain_kwargs["allowed_operators"] = (
                structured_query_translator.allowed_operators
            )
        query_constructor = load_query_constructor_runnable(
            llm,
            document_contents,
            metadata_field_info,
            enable_limit=enable_limit,
            **chain_kwargs,
        )
        query_constructor = query_constructor.with_config(
            run_name=QUERY_CONSTRUCTOR_RUN_NAME
        )
        return cls(  # type: ignore[call-arg]
            query_constructor=query_constructor,
            vectorstore=vectorstore,
            use_original_query=use_original_query,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )
