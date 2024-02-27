"""Abstract interface for document loader implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_community.document_loaders.blob_loaders import Blob


class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    A blob parser provides a way to parse raw data stored in a blob into one
    or more documents.

    The parser can be composed with blob loaders, making it easy to reuse
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


__all__ = ["BaseLoader", "BaseBlobParser"]
