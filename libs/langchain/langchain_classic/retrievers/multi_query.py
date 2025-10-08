import asyncio
import logging
from collections.abc import Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain

logger = logging.getLogger(__name__)


class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    @override
    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


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


def _unique_documents(documents: Sequence[Document]) -> list[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


class MultiQueryRetriever(BaseRetriever):
    """Given a query, use an LLM to write a set of queries.

    Retrieve docs for each query. Return the unique union of all retrieved docs.
    """

    retriever: BaseRetriever
    llm_chain: Runnable
    verbose: bool = True
    parser_key: str = "lines"
    """DEPRECATED. parser_key is no longer used and should not be specified."""
    include_original: bool = False
    """Whether to include the original query in the list of generated queries."""

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
        parser_key: str | None = None,  # noqa: ARG003
        include_original: bool = False,  # noqa: FBT001,FBT002
    ) -> "MultiQueryRetriever":
        """Initialize from llm using default template.

        Args:
            retriever: retriever to query documents from
            llm: llm for query generation using DEFAULT_QUERY_PROMPT
            prompt: The prompt which aims to generate several different versions
                of the given user query
            parser_key: DEPRECATED. `parser_key` is no longer used and should not be
                specified.
            include_original: Whether to include the original query in the list of
                generated queries.

        Returns:
            MultiQueryRetriever
        """
        output_parser = LineListOutputParser()
        llm_chain = prompt | llm | output_parser
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            include_original=include_original,
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Get relevant documents given a user query.

        Args:
            query: user query
            run_manager: the callback handler to use.

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = await self.agenerate_queries(query, run_manager)
        if self.include_original:
            queries.append(query)
        documents = await self.aretrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    async def agenerate_queries(
        self,
        question: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[str]:
        """Generate queries based upon user input.

        Args:
            question: user query
            run_manager: the callback handler to use.

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = await self.llm_chain.ainvoke(
            {"question": question},
            config={"callbacks": run_manager.get_child()},
        )
        lines = response["text"] if isinstance(self.llm_chain, LLMChain) else response
        if self.verbose:
            logger.info("Generated queries: %s", lines)
        return lines

    async def aretrieve_documents(
        self,
        queries: list[str],
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list
            run_manager: the callback handler to use

        Returns:
            List of retrieved Documents
        """
        document_lists = await asyncio.gather(
            *(
                self.retriever.ainvoke(
                    query,
                    config={"callbacks": run_manager.get_child()},
                )
                for query in queries
            ),
        )
        return [doc for docs in document_lists for doc in docs]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Get relevant documents given a user query.

        Args:
            query: user query
            run_manager: the callback handler to use.

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = self.generate_queries(query, run_manager)
        if self.include_original:
            queries.append(query)
        documents = self.retrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    def generate_queries(
        self,
        question: str,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[str]:
        """Generate queries based upon user input.

        Args:
            question: user query
            run_manager: run manager for callbacks

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = self.llm_chain.invoke(
            {"question": question},
            config={"callbacks": run_manager.get_child()},
        )
        lines = response["text"] if isinstance(self.llm_chain, LLMChain) else response
        if self.verbose:
            logger.info("Generated queries: %s", lines)
        return lines

    def retrieve_documents(
        self,
        queries: list[str],
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list
            run_manager: run manager for callbacks

        Returns:
            List of retrieved Documents
        """
        documents = []
        for query in queries:
            docs = self.retriever.invoke(
                query,
                config={"callbacks": run_manager.get_child()},
            )
            documents.extend(docs)
        return documents

    def unique_union(self, documents: list[Document]) -> list[Document]:
        """Get unique Documents.

        Args:
            documents: List of retrieved Documents

        Returns:
            List of unique retrieved Documents
        """
        return _unique_documents(documents)
