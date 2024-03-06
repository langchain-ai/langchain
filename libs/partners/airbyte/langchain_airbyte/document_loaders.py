"""Airbyte vector stores."""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
)

import airbyte as ab
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore

VST = TypeVar("VST", bound=VectorStore)


class AirbyteLoader(BaseLoader):
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
