"""Retriever that generates and executes structured queries over its own data source."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains import LLMChain
from langchain.chains.query_constructor.base import load_query_constructor_chain
from langchain.chains.query_constructor.ir import StructuredQuery, Visitor
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.retrievers.self_query.dashvector import DashvectorTranslator
from langchain.retrievers.self_query.deeplake import DeepLakeTranslator
from langchain.retrievers.self_query.elasticsearch import ElasticsearchTranslator
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain.retrievers.self_query.myscale import MyScaleTranslator
from langchain.retrievers.self_query.opensearch import OpenSearchTranslator
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain.retrievers.self_query.qdrant import QdrantTranslator
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.retrievers.self_query.supabase import SupabaseVectorTranslator
from langchain.retrievers.self_query.timescalevector import TimescaleVectorTranslator
from langchain.retrievers.self_query.vectara import VectaraTranslator
from langchain.retrievers.self_query.weaviate import WeaviateTranslator
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.vectorstores import (
    Chroma,
    DashVector,
    DeepLake,
    ElasticsearchStore,
    Milvus,
    MyScale,
    OpenSearchVectorSearch,
    Pinecone,
    Qdrant,
    Redis,
    SupabaseVectorStore,
    TimescaleVector,
    Vectara,
    VectorStore,
    Weaviate,
)

logger = logging.getLogger(__name__)


def _get_builtin_translator(vectorstore: VectorStore) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
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
        raise ValueError(
            f"Self query retriever with Vector Store type {vectorstore.__class__}"
            f" not supported."
        )


class SelfQueryRetriever(BaseRetriever, BaseModel):
    """Retriever that uses a vector store and an LLM to generate
    the vector store queries."""

    vectorstore: VectorStore
    """The underlying vector store from which documents will be retrieved."""
    llm_chain: LLMChain
    """The LLMChain for generating the vector store queries."""
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

    @root_validator(pre=True)
    def validate_translator(cls, values: Dict) -> Dict:
        """Validate translator."""
        if "structured_query_translator" not in values:
            values["structured_query_translator"] = _get_builtin_translator(
                values["vectorstore"]
            )
        return values

    def _get_structured_query(
        self, inputs: Dict[str, Any], run_manager: CallbackManagerForRetrieverRun
    ) -> StructuredQuery:
        structured_query = cast(
            StructuredQuery,
            self.llm_chain.predict(callbacks=run_manager.get_child(), **inputs),
        )
        return structured_query

    async def _aget_structured_query(
        self, inputs: Dict[str, Any], run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> StructuredQuery:
        structured_query = cast(
            StructuredQuery,
            await self.llm_chain.apredict(callbacks=run_manager.get_child(), **inputs),
        )
        return structured_query

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
        inputs = self.llm_chain.prep_inputs({"query": query})
        structured_query = self._get_structured_query(inputs, run_manager)
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
        inputs = self.llm_chain.prep_inputs({"query": query})
        structured_query = await self._aget_structured_query(inputs, run_manager)
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
        metadata_field_info: List[AttributeInfo],
        structured_query_translator: Optional[Visitor] = None,
        chain_kwargs: Optional[Dict] = None,
        enable_limit: bool = False,
        use_original_query: bool = False,
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        if structured_query_translator is None:
            structured_query_translator = _get_builtin_translator(vectorstore)
        chain_kwargs = chain_kwargs or {}

        if "allowed_comparators" not in chain_kwargs:
            chain_kwargs[
                "allowed_comparators"
            ] = structured_query_translator.allowed_comparators
        if "allowed_operators" not in chain_kwargs:
            chain_kwargs[
                "allowed_operators"
            ] = structured_query_translator.allowed_operators
        llm_chain = load_query_constructor_chain(
            llm,
            document_contents,
            metadata_field_info,
            enable_limit=enable_limit,
            **chain_kwargs,
        )
        return cls(
            llm_chain=llm_chain,
            vectorstore=vectorstore,
            use_original_query=use_original_query,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )
