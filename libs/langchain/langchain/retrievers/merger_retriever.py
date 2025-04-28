import asyncio

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MergerRetriever(BaseRetriever):
    """Retriever that merges the results of multiple retrievers."""

    retrievers: list[BaseRetriever]
    """A list of retrievers to merge."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = self.merge_documents(query, run_manager)

        return merged_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = await self.amerge_documents(query, run_manager)

        return merged_documents

    def merge_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            retriever.invoke(
                query,
                config={"callbacks": run_manager.get_child(f"retriever_{i + 1}")},
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Merge the results of the retrievers.
        merged_documents = []
        max_docs = max(map(len, retriever_docs), default=0)
        for i in range(max_docs):
            for retriever, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents

    async def amerge_documents(
        self, query: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Asynchronously merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = await asyncio.gather(
            *(
                retriever.ainvoke(
                    query,
                    config={"callbacks": run_manager.get_child(f"retriever_{i + 1}")},
                )
                for i, retriever in enumerate(self.retrievers)
            )
        )

        # Merge the results of the retrievers.
        merged_documents = []
        max_docs = max(map(len, retriever_docs), default=0)
        for i in range(max_docs):
            for retriever, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents
