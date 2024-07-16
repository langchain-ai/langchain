"""Airbyte vector stores."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    TypeVar,
)

import airbyte as ab
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

VST = TypeVar("VST", bound=VectorStore)


class AirbyteLoader:
    """Airbyte Document Loader.

    Example:
        .. code-block:: python

            from langchain_airbyte import AirbyteLoader

            loader = AirbyteLoader(
                source="github",
                stream="pull_requests",
            )
            documents = loader.lazy_load()
    """

    def __init__(
        self,
        source: str,
        stream: str,
        *,
        config: Optional[Dict] = None,
        include_metadata: bool = True,
        template: Optional[PromptTemplate] = None,
    ):
        self._airbyte_source = ab.get_source(source, config=config, streams=[stream])
        self._stream = stream
        self._template = template
        self._include_metadata = include_metadata

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

        if text_splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError as e:
                raise ImportError(
                    "Unable to import from langchain_text_splitters. Please specify "
                    "text_splitter or install langchain_text_splitters with "
                    "`pip install -U langchain-text-splitters`."
                ) from e
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.lazy_load()
        return _text_splitter.split_documents(docs)

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        # if no prompt template defined, use default airbyte documents
        if not self._template:
            for document in self._airbyte_source.get_documents(self._stream):
                # convert airbyte document to langchain document
                metadata = (
                    {}
                    if not self._include_metadata
                    else {
                        **document.metadata,
                        "_last_modified": document.last_modified,
                        "_id": document.id,
                    }
                )
                yield Document(
                    page_content=document.content,
                    metadata=metadata,
                )
        else:
            records: Iterator[Mapping[str, Any]] = self._airbyte_source.get_records(
                self._stream
            )
            for record in records:
                metadata = {} if not self._include_metadata else dict(record)
                yield Document(
                    page_content=self._template.format(**record), metadata=metadata
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
