"""Retriever that wraps a base retriever and filters the results."""
from typing import List

from pydantic import BaseModel, Extra

from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.schema import BaseRetriever, Document


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """Retriever that wraps a base retriever and compresses the results."""

    base_compressor: BaseDocumentCompressor
    """Compressor for compressing retrieved documents."""

    base_retriever: BaseRetriever
    """Base Retriever to use for getting relevant documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        if self.base_retriever.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.base_retriever.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.base_retriever.search_kwargs
                )
            )
            docs = [doc for doc, score in docs_and_similarities]
            compressed_docs = self.base_compressor.compress_documents(docs, query)
            compressed_docs_with_scores = zip(
                compressed_docs, [score for doc, score in docs_and_similarities]
            )
            return list(compressed_docs_with_scores)

        docs = self.base_retriever.get_relevant_documents(query)
        if docs:
            compressed_docs = self.base_compressor.compress_documents(docs, query)
            return list(compressed_docs)
        else:
            return []

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        if self.base_retriever.search_type == "similarity_score_threshold":
            docs_and_similarities = await self.base_retriever.vectorstore.asimilarity_search_with_relevance_scores(
                query, **self.base_retriever.search_kwargs
            )
            docs = [doc for doc, score in docs_and_similarities]
            compressed_docs = await self.base_compressor.acompress_documents(
                docs, query
            )
            compressed_docs_with_scores = zip(
                compressed_docs, [score for doc, score in docs_and_similarities]
            )
            return list(compressed_docs_with_scores)

        docs = await self.base_retriever.aget_relevant_documents(query)
        if docs:
            compressed_docs = await self.base_compressor.acompress_documents(
                docs, query
            )
            return list(compressed_docs)
        else:
            return []
