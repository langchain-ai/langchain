from typing import Any, Dict, List, Optional, Type, cast

from pydantic import BaseModel, Field, root_validator

from langchain import LLMChain
from langchain.chains.query_constructor.base import (
    AttributeInfo,
    load_query_constructor_chain,
)
from langchain.chains.query_constructor.query_language import StructuredQuery, Visitor
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain.schema import BaseLanguageModel, BaseRetriever, Document
from langchain.vectorstores import Pinecone, VectorStore


def get_builtin_translator(vectorstore_cls: Type[VectorStore]) -> Visitor:
    """"""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
        Pinecone: PineconeTranslator
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
    """"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_translator(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "structured_query_translator" not in values:
            vectorstore_cls = values["vectorstore"].__class__
            values["structured_query_translator"] = get_builtin_translator(
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
        inputs = self.llm_chain.prep_inputs(query)
        structured_query = cast(
            StructuredQuery, self.llm_chain.predict_and_parse(**inputs)
        )
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query
        )
        search_kwargs = {**self.search_kwargs, **new_kwargs}
        docs = self.vectorstore.search(query, self.search_type, **search_kwargs)
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
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        if structured_query_translator is None:
            structured_query_translator = get_builtin_translator(vectorstore.__class__)
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
            llm, document_contents, metadata_field_info, **chain_kwargs
        )
        return cls(
            llm_chain=llm_chain,
            vectorstore=vectorstore,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )
