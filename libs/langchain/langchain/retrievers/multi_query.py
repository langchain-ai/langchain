import asyncio
import logging
from typing import Any, List, Optional, Sequence, cast

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains.llm import LLMChain
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.retrievers.rrf import weighted_reciprocal_rank_fusion
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel

logger = logging.getLogger(__name__)


class LineList(BaseModel):
    """List of lines."""

    lines: List[str] = Field(description="Lines of text")
    """List of lines."""


class LineListOutputParser(PydanticOutputParser):
    """Output parser for a list of lines."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


# Default prompt
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
    to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}""",
)


def _unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


class MultiQueryRetriever(BaseRetriever):
    """Given a query, use an LLM to write a set of queries.

    Retrieve docs for each query. Return the unique union of all retrieved docs.
    """

    retriever: BaseRetriever
    """Retriever to query documents"""
    llm_chain: LLMChain
    """Chain to generate query variations"""
    verbose: bool = True
    """Set verbosity to log generated queries"""
    parser_key: str = "lines"
    """Key to access the generated queries in the llm_chain's output parser"""

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt: PromptTemplate = DEFAULT_QUERY_PROMPT,
        **kwargs: Any,
    ) -> "MultiQueryRetriever":
        """Initialize from llm using default template.

        Args:
            retriever: retriever to query documents from
            llm: llm for query generation
            prompt: prompt for query generation. Defaults to DEFAULT_QUERY_PROMPT.

        Returns:
            MultiQueryRetriever
        """
        output_parser = LineListOutputParser()
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=output_parser,
            verbose=kwargs.get("verbose", False),
        )
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            **kwargs,
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            query: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = await self.agenerate_queries(query, run_manager)
        doc_lists = await self.aretrieve_documents(queries, run_manager)
        documents = [doc for docs in doc_lists for doc in docs]
        return self.unique_union(documents)

    async def agenerate_queries(
        self, question: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = await self.llm_chain.acall(
            inputs={"question": question}, callbacks=run_manager.get_child()
        )
        lines = getattr(response["text"], self.parser_key, [])
        if self.verbose:
            logger.info(f"Generated queries: {lines}")
        return lines

    async def aretrieve_documents(
        self, queries: List[str], run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[List[Document]]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        document_lists = await asyncio.gather(
            *(
                self.retriever.aget_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
                for query in queries
            )
        )
        document_lists = cast(List[List[Document]], document_lists)
        return document_lists

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            query: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = self.generate_queries(query, run_manager)
        doc_lists = self.retrieve_documents(queries, run_manager)
        documents = [doc for docs in doc_lists for doc in docs]
        return self.unique_union(documents)

    def generate_queries(
        self, question: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = self.llm_chain(
            {"question": question}, callbacks=run_manager.get_child()
        )
        lines = getattr(response["text"], self.parser_key, [])
        if self.verbose:
            logger.info(f"Generated queries: {lines}")
        return lines

    def retrieve_documents(
        self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
    ) -> List[List[Document]]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
            documents.append(docs)
        return documents

    def unique_union(self, documents: List[Document]) -> List[Document]:
        """Get unique Documents.

        Args:
            documents: List of retrieved Documents

        Returns:
            List of unique retrieved Documents
        """
        return _unique_documents(documents)


class MultiQueryRankFusionRetriever(MultiQueryRetriever):
    """
    Given a query, use an LLM to write a set of query variations.
    Then, retrieve docs for each query, and return the retrieved docs
    re-ranked with weighted Reciprocal Rank Fusion (RRF).
    """

    append_original_query: bool = True
    """Whether to append to original query to the variations generated by the chain"""
    weights: Optional[List[float]] = None
    """Weights assigned to the different queries for rerank. 
    They can be used to assign more weight to the original query."""
    c: int = 60
    """Constant added to the rank, controlling the balance between
    the importance of high-ranked items and the consideration given to
    lower-ranked items"""
    k: Optional[int] = None
    """Top k Documents to return after RRF. If None, all Documents are returned."""

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            query: user query

        Returns:
            List of relevant documents from all generated queries, re-ranked with RRF.
        """
        queries = await self.agenerate_queries(query, run_manager)

        if self.append_original_query and query not in queries:
            queries.append(query)

        doc_lists = await self.aretrieve_documents(queries, run_manager)
        documents = self.rank_fusion(doc_lists)

        if self.k is not None:
            documents = documents[: self.k]

        return documents

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            query: user query

        Returns:
            List of relevant documents from all generated queries, re-ranked with RRF.
        """
        queries = self.generate_queries(query, run_manager)

        if self.append_original_query and query not in queries:
            queries.append(query)

        doc_lists = self.retrieve_documents(queries, run_manager)
        documents = self.rank_fusion(doc_lists)

        if self.k is not None:
            documents = documents[: self.k]

        return documents

    def rank_fusion(self, doc_lists: List[List[Document]]) -> List[Document]:
        """Get unique Documents.

        Args:
            doc_lists: List of Document lists, one per query

        Returns:
            List of relevant documents from all generated queries, re-ranked with RRF.
        """
        return weighted_reciprocal_rank_fusion(
            doc_lists, weights=self.weights, c=self.c
        )
