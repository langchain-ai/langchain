"""Interface for retrieved document filters."""
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field

from langchain.schema import Document


class _RetrievedDocument(Document):
    """Wrapper for a retrieved document that includes metadata about the query."""

    query_metadata: dict = Field(default_factory=dict)
    """Metadata associated with the query for which the document was retrieved."""

    def to_document(self) -> Document:
        """Convert the RetrievedDocument to a Document."""
        return Document(page_content=self.page_content, metadata=self.metadata)

    @classmethod
    def from_document(cls, doc: Document) -> "_RetrievedDocument":
        """Create a RetrievedDocument from a Document."""
        return cls(page_content=doc.page_content, metadata=doc.metadata)


class BaseDocumentCompressor(BaseModel, ABC):
    """Interface for retrieved document compressors."""

    @abstractmethod
    def compress_documents(
        self, documents: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Compress retrieved documents given the query context."""

    @abstractmethod
    async def acompress_documents(
        self, documents: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Compress retrieved documents given the query context."""
