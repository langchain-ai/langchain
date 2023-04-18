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


class BaseDocumentFilter(BaseModel, ABC):
    """Interface for retrieved document filters."""

    @abstractmethod
    def filter(
        self, docs: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Filter down documents."""

    @abstractmethod
    async def afilter(
        self, docs: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Filter down documents."""
