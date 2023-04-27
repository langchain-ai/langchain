"""Abstract interface for document loader implementations."""

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class BaseLoader(ABC):
    """Interface for loading documents.

    Implementations should implement the lazy-loading method using generators
    to avoid loading all documents into memory at once.

    The `load` method will remain as is for backwards compatibility, but it's
    implementation should be just `list(self.lazy_load())`.
    """

    # Sub-classes should implement this method
    # as return list(self.lazy_load()).
    # This method returns a List which is materialized in memory.
    @abstractmethod
    def load(self) -> List[Document]:
        """Load data into document objects."""

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """Load documents and split into chunks."""
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.load()
        return _text_splitter.split_documents(docs)

    # Attention: This method will be upgraded into an abstractmethod once it's
    #            implemented in all the existing subclasses.
    def lazy_load(
        self,
    ) -> Iterable[Document]:
        """A lazy loader for document content."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )
