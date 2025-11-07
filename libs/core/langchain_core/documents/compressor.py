"""Document compressor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import Document


class BaseDocumentCompressor(BaseModel, ABC):
    """Base class for document compressors.

    This abstraction is primarily used for post-processing of retrieved documents.

    `Document` objects matching a given query are first retrieved.

    Then the list of documents can be further processed.

    For example, one could re-rank the retrieved documents using an LLM.

    !!! note
        Users should favor using a `RunnableLambda` instead of sub-classing from this
        interface.

    """

    @abstractmethod
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context.

        Args:
            documents: The retrieved `Document` objects.
            query: The query context.
            callbacks: Optional `Callbacks` to run during compression.

        Returns:
            The compressed documents.

        """

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Async compress retrieved documents given the query context.

        Args:
            documents: The retrieved `Document` objects.
            query: The query context.
            callbacks: Optional `Callbacks` to run during compression.

        Returns:
            The compressed documents.

        """
        return await run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )
