"""Airbyte vector stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Dict, Iterator, List, Optional, TypeVar

import airbyte as ab
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain.text_splitter import TextSplitter
    from langchain_core.documents import Document

VST = TypeVar("VST", bound=VectorStore)


class AirbyteLoader:
    """Airbyte Document Loader.

    Example:
        .. code-block:: python

            from langchain_airbyte import AirbyteLoader

            loader = AirbyteLoader(
                source="github",

            )
            documents = loader.lazy_load()
    """

    def __init__(
        self,
        source: str,
        *,
        config: Optional[Dict] = None,
        streams: Optional[List[str]] = None,
    ):
        self._airbyte_source = ab.get_source(source, config=config, streams=streams)

    def load(self) -> List[Document]:
        """Load source data into Document objects."""
        return list(self.lazy_load())

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """Load Documents and split into chunks. Chunks are returned as Documents.

        Args:
            text_splitter: TextSplitter instance to use for splitting documents.
              Defaults to RecursiveCharacterTextSplitter.

        Returns:
            List of Documents.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.load()
        return _text_splitter.split_documents(docs)

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """A lazy loader for Documents."""
        iterator = await run_in_executor(None, self.lazy_load)
        done = object()
        while True:
            doc = await run_in_executor(None, next, iterator, done)  # type: ignore[call-arg, arg-type]
            if doc is done:
                break
            yield doc  # type: ignore[misc]
