from typing import Any, Callable, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field, root_validator

from langchain import LLMChain
from langchain.chains.query_constructor.base import (
    AttributeInfo,
    Comparison,
    Operation,
    load_query_constructor_chain,
)
from langchain.llms import BaseLLM
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import VectorStore


class SelfQueryRetriever(BaseRetriever, BaseModel):
    """Retriever that wraps around a vector store and uses an LLM to generate
    the vector store queries."""

    vectorstore: VectorStore
    """The Pinecone vector store from which documents will be retrieved."""
    llm_chain: LLMChain
    """The LLMChain for generating the vector store queries."""
    search_type: str = "similarity"
    """The search type to perform on the vector store."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass in to the vector store search."""
    structured_query_to_filter: Callable[[Optional[Union[Comparison, Operation]]], dict]
    """"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "mmr"):
                raise ValueError(
                    f"search_type of {search_type} not allowed. Expected "
                    "search_type to be 'similarity' or 'mmr'."
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
        structured_query = cast(dict, self.llm_chain.predict_and_parse(**inputs))
        print(structured_query)
        new_query = structured_query["query"]
        new_filter = self.structured_query_to_filter(structured_query["filter"])
        print(new_filter)
        search_kwargs = {k: v for k, v in self.search_kwargs.items() if k != "filter"}
        docs = self.vectorstore.search(
            new_query, self.search_type, filter=new_filter, **search_kwargs
        )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        document_contents: str,
        metadata_field_info: List[AttributeInfo],
        chain_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        chain_kwargs = chain_kwargs or {}
        llm_chain = load_query_constructor_chain(
            llm, document_contents, metadata_field_info, **chain_kwargs
        )
        return cls(
            llm_chain=llm_chain,
            vectorstore=vectorstore,
            **kwargs,
        )
