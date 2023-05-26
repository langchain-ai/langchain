"""Retriever that generates and executes structured queries over its own data source."""
from typing import Any, Dict, List, Optional, Type, cast

from pydantic import BaseModel, Field, root_validator

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.chains.query_constructor.base import load_query_constructor_chain
from langchain.chains.query_constructor.ir import StructuredQuery, Visitor
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain.retrievers.self_query.weaviate import WeaviateTranslator
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import Chroma, Pinecone, VectorStore, Weaviate


def _get_builtin_translator(vectorstore_cls: Type[VectorStore]) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
        Pinecone: PineconeTranslator,
        Chroma: ChromaTranslator,
        Weaviate: WeaviateTranslator,
    }
    if vectorstore_cls not in BUILTIN_TRANSLATORS:
        raise ValueError(
            f"Self query retriever with Vector Store type {vectorstore_cls}"
            f" not supported."
        )
    return BUILTIN_TRANSLATORS[vectorstore_cls]()


class SelfQueryRetriever(BaseRetriever, BaseModel):
    """Retriever that wraps around a vector store and uses an LLM to generate
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

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_translator(cls, values: Dict) -> Dict:
        """Validate translator."""
        if "structured_query_translator" not in values:
            vectorstore_cls = values["vectorstore"].__class__
            values["structured_query_translator"] = _get_builtin_translator(
                vectorstore_cls
            )
        return values

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        inputs = self.llm_chain.prep_inputs({"query": query})
        structured_query = cast(
            StructuredQuery, self.llm_chain.predict_and_parse(callbacks=None, **inputs)
        )
        if self.verbose:
            print(structured_query)
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query
        )
        if structured_query.limit is not None:
            new_kwargs["k"] = structured_query.limit

        search_kwargs = {**self.search_kwargs, **new_kwargs}
        docs = self.vectorstore.search(new_query, self.search_type, **search_kwargs)
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

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
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        if structured_query_translator is None:
            structured_query_translator = _get_builtin_translator(vectorstore.__class__)
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
            structured_query_translator=structured_query_translator,
            **kwargs,
        )
