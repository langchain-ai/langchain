from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.schema.text_splitter import TextSplitterInterface


class DocumentLoaderInterface(ABC):
    """Interface for Document Loader.

    Implementations should implement the lazy-loading method using generators
    to avoid loading all Documents into memory at once.

    The `load` method will remain as is for backwards compatibility, but its
    implementation should be just `list(self.lazy_load())`.
    """

    @abstractmethod
    def load(self) -> List[Document]:
        """Load data into Document objects.

        Sub-classes should implement this method
        as `return list(self.lazy_load())`
        This method returns a List which is materialized in memory.
        """

    @abstractmethod
    def load_and_split(
        self, text_splitter: Optional[TextSplitterInterface] = None
    ) -> List[Document]:
        """Load Documents and split into chunks. Chunks are returned as Documents.

        Args:
            text_splitter: TextSplitter instance to use for splitting documents.
              Defaults to RecursiveCharacterTextSplitter.

        Returns:
            List of Documents.
        """

    # Attention: This method will be upgraded into an abstractmethod once it's
    #            implemented in all the existing subclasses.
    @abstractmethod
    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents."""
