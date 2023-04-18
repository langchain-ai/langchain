"""Retriever that wraps a base retriever and filters the results."""
from typing import List

from pydantic import BaseModel, Extra

from langchain.retrievers.document_filters.base import (
    BaseDocumentFilter,
    _RetrievedDocument,
)
from langchain.schema import BaseRetriever, Document


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """Retriever that wraps a base retriever and filters the results."""

    base_filter: BaseDocumentFilter
    """Filter for filtering documents."""

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
        docs = self.base_retriever.get_relevant_documents(query)
        retrieved_docs = [_RetrievedDocument.from_document(doc) for doc in docs]
        compressed_docs = self.base_filter.filter(retrieved_docs, query)
        return [rdoc.to_document() for rdoc in compressed_docs]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.base_retriever.aget_relevant_documents(query)
        retrieved_docs = [_RetrievedDocument.from_document(doc) for doc in docs]
        compressed_docs = await self.base_filter.afilter(retrieved_docs, query)
        return [rdoc.to_document() for rdoc in compressed_docs]
