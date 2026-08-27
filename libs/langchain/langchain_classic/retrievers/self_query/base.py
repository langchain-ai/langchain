"""Retriever that generates and executes structured queries over its own data source."""

import logging
from collections.abc import Sequence
from typing import Any

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
from typing_extensions import override

from langchain_classic.chains.query_constructor.base import (
    load_query_constructor_runnable,
)
from langchain_classic.chains.query_constructor.schema import AttributeInfo

logger = logging.getLogger(__name__)
QUERY_CONSTRUCTOR_RUN_NAME = "query_constructor"


def _get_builtin_translator(vectorstore: VectorStore) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    try:
        import langchain_community  # noqa: F401
    except ImportError as err:
        msg = (
            "The langchain-community package must be installed to use this feature."
            " Please install it using `pip install langchain-community`."
        )
        raise ImportError(msg) from err

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

    builtin_translators: dict[type[VectorStore], type[Visitor]] = {
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
    if isinstance(vectorstore, MyScale):
        return MyScaleTranslator(metadata_key=vectorstore.metadata_column)
    if isinstance(vectorstore, Redis):
        return RedisTranslator.from_vectorstore(vectorstore)
    if isinstance(vectorstore, TencentVectorDB):
        fields = [
            field.name for field in (vectorstore.meta_fields or []) if field.index
        ]
        return TencentVectorDBTranslator(fields)
    if vectorstore.__class__ in builtin_translators:
        return builtin_translators[vectorstore.__class__]()
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
        from langchain_milvus import Milvus
    except ImportError:
        pass
    else:
        if isinstance(vectorstore, Milvus):
            return MilvusTranslator()

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
        # Trying langchain_chroma import if exists
        from langchain_chroma import Chroma
    except ImportError:
        pass
    else:
        if isinstance(vectorstore, Chroma):
            return ChromaTranslator()

    try:
        from langchain_postgres import PGVector
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

    try:
        # Trying langchain_weaviate (weaviate v4) import if exists
        from langchain_weaviate.vectorstores import WeaviateVectorStore

    except ImportError:
        pass
    else:
        if isinstance(vectorstore, WeaviateVectorStore):
            return WeaviateTranslator()

    msg = (
        f"Self query retriever with Vector Store type {vectorstore.__class__}"
        f" not supported."
    )
    raise ValueError(msg)


class SelfQueryRetriever(BaseRetriever):
    """Self Query Retriever.

    Retriever that uses a vector store and an LLM to generate the vector store queries.
    """

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
    """Translator for turning internal query language into `VectorStore` search params."""  # noqa: E501
    verbose: bool = False

    use_original_query: bool = False
    """Use original query instead of the revised new query from LLM"""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_translator(cls, values: dict) -> Any:
        """Validate translator."""
        if "structured_query_translator" not in values:
            values["structured_query_translator"] = _get_builtin_translator(
                values["vectorstore"],
            )
        return values

    @property
    def llm_chain(self) -> Runnable:
        """llm_chain is legacy name kept for backwards compatibility."""
        return self.query_constructor

    def _prepare_query(
        self,
        query: str,
        structured_query: StructuredQuery,
    ) -> tuple[str, dict[str, Any]]:
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query,
        )
        if structured_query.limit is not None:
            new_kwargs["k"] = structured_query.limit
        if self.use_original_query:
            new_query = query
        search_kwargs = {**self.search_kwargs, **new_kwargs}
        return new_query, search_kwargs

    def _get_docs_with_query(
        self,
        query: str,
        search_kwargs: dict[str, Any],
    ) -> list[Document]:
        return self.vectorstore.search(query, self.search_type, **search_kwargs)

    async def _aget_docs_with_query(
        self,
        query: str,
        search_kwargs: dict[str, Any],
    ) -> list[Document]:
        return await self.vectorstore.asearch(query, self.search_type, **search_kwargs)

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        structured_query = self.query_constructor.invoke(
            {"query": query},
            config={"callbacks": run_manager.get_child()},
        )
        if self.verbose:
            logger.info("Generated Query: %s", structured_query)
        new_query, search_kwargs = self._prepare_query(query, structured_query)
        return self._get_docs_with_query(new_query, search_kwargs)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        structured_query = await self.query_constructor.ainvoke(
            {"query": query},
            config={"callbacks": run_manager.get_child()},
        )
        if self.verbose:
            logger.info("Generated Query: %s", structured_query)
        new_query, search_kwargs = self._prepare_query(query, structured_query)
        return await self._aget_docs_with_query(new_query, search_kwargs)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        document_contents: str,
        metadata_field_info: Sequence[AttributeInfo | dict],
        structured_query_translator: Visitor | None = None,
        chain_kwargs: dict | None = None,
        enable_limit: bool = False,  # noqa: FBT001,FBT002
        use_original_query: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        """Create a SelfQueryRetriever from an LLM and a vector store.

        Args:
            llm: The language model to use for generating queries.
            vectorstore: The vector store to use for retrieving documents.
            document_contents: Description of the page contents of the document to be
                queried.
            metadata_field_info: Metadata field information for the documents.
            structured_query_translator: Optional translator for turning internal query
                language into `VectorStore` search params.
            chain_kwargs: Additional keyword arguments for the query constructor.
            enable_limit: Whether to enable the limit operator.
            use_original_query: Whether to use the original query instead of the revised
                query from the LLM.
            **kwargs: Additional keyword arguments for the SelfQueryRetriever.

        Returns:
            An instance of SelfQueryRetriever.
        """
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
            run_name=QUERY_CONSTRUCTOR_RUN_NAME,
        )
        return cls(
            query_constructor=query_constructor,
            vectorstore=vectorstore,
            use_original_query=use_original_query,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )
