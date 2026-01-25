"""Abstract interface for document loader implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_text_splitters import TextSplitter

    from langchain_core.documents import Document
    from langchain_core.documents.base import Blob

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _HAS_TEXT_SPLITTERS = True
except ImportError:
    _HAS_TEXT_SPLITTERS = False


class BaseLoader(ABC):  # noqa: B024
    """Interface for document loader.

    Implementations should implement the lazy-loading method using generators to avoid
    loading all documents into memory at once.

    `load` is provided just for user convenience and should not be overridden.
    """

    # Sub-classes should not implement this method directly. Instead, they
    # should implement the lazy load method.
    def load(self) -> list[Document]:
        """Load data into `Document` objects.

        Returns:
            The documents.
        """
        return list(self.lazy_load())

    async def aload(self) -> list[Document]:
        """Load data into `Document` objects.

        Returns:
            The documents.
        """
        return [document async for document in self.alazy_load()]

    def load_and_split(
        self, text_splitter: TextSplitter | None = None
    ) -> list[Document]:
        """Load `Document` and split into chunks. Chunks are returned as `Document`.

        !!! danger

            Do not override this method. It should be considered to be deprecated!

        Args:
            text_splitter: `TextSplitter` instance to use for splitting documents.

                Defaults to `RecursiveCharacterTextSplitter`.

        Raises:
            ImportError: If `langchain-text-splitters` is not installed and no
                `text_splitter` is provided.

        Returns:
            List of `Document` objects.
        """
        if text_splitter is None:
            if not _HAS_TEXT_SPLITTERS:
                msg = (
                    "Unable to import from langchain_text_splitters. Please specify "
                    "text_splitter or install langchain_text_splitters with "
                    "`pip install -U langchain-text-splitters`."
                )
                raise ImportError(msg)

            text_splitter_: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            text_splitter_ = text_splitter
        docs = self.load()
        return text_splitter_.split_documents(docs)

    # Attention: This method will be upgraded into an abstractmethod once it's
    #            implemented in all the existing subclasses.
    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for `Document`.

        Yields:
            The `Document` objects.
        """
        if type(self).load != BaseLoader.load:
            return iter(self.load())
        msg = f"{self.__class__.__name__} does not implement lazy_load()"
        raise NotImplementedError(msg)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """A lazy loader for `Document`.

        Yields:
            The `Document` objects.
        """
        iterator = await run_in_executor(None, self.lazy_load)
        done = object()
        while True:
            doc = await run_in_executor(None, next, iterator, done)
            if doc is done:
                break
            yield doc  # type: ignore[misc]


class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    A blob parser provides a way to parse raw data stored in a blob into one or more
    `Document` objects.

    The parser can be composed with blob loaders, making it easy to reuse a parser
    independent of how the blob was originally loaded.
    """

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazy parsing interface.

        Subclasses are required to implement this method.

        Args:
            blob: `Blob` instance

        Returns:
            Generator of `Document` objects
        """

    def parse(self, blob: Blob) -> list[Document]:
        """Eagerly parse the blob into a `Document` or list of `Document` objects.

        This is a convenience method for interactive development environment.

        Production applications should favor the `lazy_parse` method instead.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: `Blob` instance

        Returns:
            List of `Document` objects
        """
        return list(self.lazy_parse(blob))
