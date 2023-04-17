""""""
from typing import List

from pydantic import BaseModel, Extra

from langchain.document_filter.base import BaseDocumentFilter
from langchain.schema import BaseRetriever, Document


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """"""

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
            List of relevant documents
        """
        docs = self.base_retriever.get_relevant_documents(query)
        compressed_docs = self.base_filter.filter(docs, query)
        return compressed_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.base_retriever.aget_relevant_documents(query)
        compressed_docs = self.base_filter.afilter(docs, query)
        return compressed_docs
