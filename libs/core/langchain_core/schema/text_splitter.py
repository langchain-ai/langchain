from abc import ABC, abstractmethod
from typing import (
    Iterable,
    List,
    Optional,
)

from langchain_core.documents import Document


# TODO: this is a work in progress
# TODO: do we need to derive from DocumentTransformer?
class TextSplitterInterface(ABC):
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
