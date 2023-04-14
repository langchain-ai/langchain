"""Base loader class."""
from __future__ import annotations

import abc
from abc import ABC, abstractmethod
from typing import List, Optional, Generator

from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class BaseLoader(ABC):
    """Base loader for documents."""

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

    @abstractmethod
    def lazy_load(
        self,
    ) -> Generator[Document, None, None]:
        """A lazy loader for document content."""
        raise NotImplementedError()


class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    * A blob is a representation of raw data
    * A blob parser provides a way to parse a blob into one or more documents
    """

    @abc.abstractmethod
    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Lazy parsing interface.

        Subclasses should implement this method and

        Args:
            blob: Blob instance

        Returns:
            Generator of documents
        """
        raise NotImplementedError()

    def parse(self, blob: Blob) -> List[Document]:
        """Eagerly parse the blob into a document or documents.

        This is a convenience method when prototyping interactively.

        For serious use, the lazy_parse method should be used instead as it allows
        for lazy loading of content.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: Blob instance

        Returns:
            List of documents
        """
        return list(self.lazy_parse(blob))
