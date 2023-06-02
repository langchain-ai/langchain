"""Abstract interface for document loader implementations."""
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class BaseLoader(ABC):
    """Interface for loading documents.

    Implementations should implement the lazy-loading method using generators
    to avoid loading all documents into memory at once.

    The `load` method will remain as is for backwards compatibility, but its
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
    ) -> Iterator[Document]:
        """A lazy loader for document content."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )


class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    A blob parser provides a way to parse raw data stored in a blob into one
    or more documents.

    The parser can be composed with blob loaders, making it easy to re-use
    a parser independent of how the blob was originally loaded.
    """

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazy parsing interface.

        Subclasses are required to implement this method.

        Args:
            blob: Blob instance

        Returns:
            Generator of documents
        """

    def parse(self, blob: Blob) -> List[Document]:
        """Eagerly parse the blob into a document or documents.

        This is a convenience method for interactive development environment.

        Production applications should favor the lazy_parse method instead.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: Blob instance

        Returns:
            List of documents
        """
        return list(self.lazy_parse(blob))


class BaseBlobTransformer(ABC):
    """Abstract interface for blob transformers.

    A blob transformer provides a way to transform raw data stored in a blob into a
    different data format.

    The transformer can be composed with blob loaders, making it easy to re-use
    a transformer independent of how the blob was originally loaded.
    """

    @abstractmethod
    def lazy_transform(self, blob: Blob) -> Blob:
        """Lazy transformer interface.

        Subclasses are required to implement this method.

        Args:
            blob: Blob instance

        Returns:
            Blob
        """

    def transform(self, blob: Blob) -> Blob:
        """Eagerly transform the blob into a different type.

        This is a convenience method for interactive development environment.

        Production applications should favor the lazy_parse method instead.

        Subclasses should generally not over-ride this transform method.

        Args:
            blob: Blob instance

        Returns:
            Blob
        """
        return self.lazy_transform(blob)


class BaseBlobSplitter(ABC):
    """Abstract interface for blob splitters.

    A blob splitter provides a way to transform raw data stored in a blob into a
    different data format.

    The transformer can be composed with blob loaders, making it easy to re-use
    a transformer independent of how the blob was originally loaded.
    """

    @abstractmethod
    def lazy_split(self, blob: Blob) -> Iterator[Blob]:
        """Lazy split interface.

        Subclasses are required to implement this method.

        Args:
            blob: Blob instance

        Returns:
            Generator of blobs
        """

    def split_blob(self, blob: Blob) -> List[Blob]:
        """Eagerly split the blob into a list of blobs.

        This is a convenience method for interactive development environment.

        Production applications should favor the lazy_split method instead.

        Subclasses should generally not over-ride this split method.

        Args:
            blob: Blob instance

        Returns:
            List of blobs
        """
        return list(self.lazy_split(blob))
