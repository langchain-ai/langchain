from abc import ABC, abstractmethod
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
)

from langchain_core.documents import BaseDocumentTransformer, Document


class TextSplitterInterface(
    BaseDocumentTransformer, ABC
):  # TODO Do we need ABC since we have BaseDocumentTransformer?
    """Interface for splitting text into chunks."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    @abstractmethod
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""

    @abstractmethod
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
