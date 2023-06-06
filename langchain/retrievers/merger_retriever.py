from langchain.schema import BaseRetriever
from langchain.schema import BaseDocumentTransformer, Document
from typing import List
import asyncio


class MergerRetriever(BaseRetriever):
    """
    This class merges the results of multiple retrievers and then refines the results using a list of transformers.

    Args:
        retrievers: A list of retrievers to merge.
        transformers: A list of transformers to refine the results.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        transformers: List[BaseDocumentTransformer],
    ):
        """
        Initialize the MergerRetriever class.

        Args:
            retrievers: A list of retrievers to merge.
            transformers: A list of transformers to refine the results.
        """

        self.retrievers = retrievers
        self.transformers = transformers

    def get_relevant_documents(self, query: str) -> List[str]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = self.merge_documents(query)

        # Refine the results using the transformers.
        refined_documents = self.refine_documents(merged_documents)

        return refined_documents

    async def aget_relevant_documents(self, query: str) -> List[str]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = await self.amerge_documents(query)

        # Refine the results using the transformers.
        refined_documents = await self.arefine_documents(merged_documents)

        return refined_documents

    def merge_documents(self, query: str) -> List[str]:
        """
        Merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            retriever.get_relevant_documents(query) for retriever in self.retrievers
        ]

        # Merge the results of the retrievers.
        merged_documents = []
        max_docs = max(len(docs) for docs in retriever_docs)
        for i in range(max_docs):
            for retriever, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents

    async def amerge_documents(self, query: str) -> List[str]:
        """
        Asynchronously merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            await retriever.aget_relevant_documents(query)
            for retriever in self.retrievers
        ]

        # Merge the results of the retrievers.
        merged_documents = []
        max_docs = max(len(docs) for docs in retriever_docs)
        for i in range(max_docs):
            for retriever, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents

    def refine_documents(self, documents: List[str]) -> List[str]:
        """
        Refine the results of the retrievers using a list of transformers.

        Args:
            documents: The documents to refine.

        Returns:
            A list of refined documents.
        """

        # Create a list to store the refined documents.
        refined_documents = documents

        # Iterate over the transformers and refine the documents.
        for transformer in self.transformers:
            refined_documents = transformer.transform_documents(refined_documents)

        return refined_documents

    async def arefine_documents(self, documents: List[str]) -> List[str]:
        """
        Asynchronously refine the results of the retrievers using a list of transformers.

        Args:
            documents: The documents to refine.

        Returns:
            A list of refined documents.
        """

        # Create a list to store the refined documents.
        refined_documents = documents

        # Iterate over the transformers and refine the documents.
        for transformer in self.transformers:
            if hasattr(transformer, "atransform_documents"):
                refined_documents = await transformer.atransform_documents(
                    refined_documents
                )

        return refined_documents
