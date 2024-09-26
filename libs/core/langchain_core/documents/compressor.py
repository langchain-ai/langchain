from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.runnables import run_in_executor


class BaseDocumentCompressor(BaseModel, ABC):
    """Base class for document compressors.

    This abstraction is primarily used for
    post-processing of retrieved documents.

    Documents matching a given query are first retrieved.
    Then the list of documents can be further processed.

    For example, one could re-rank the retrieved documents
    using an LLM.

    **Note** users should favor using a RunnableLambda
    instead of sub-classing from this interface.
    """

    @abstractmethod
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context.

        Args:
            documents: The retrieved documents.
            query: The query context.
            callbacks: Optional callbacks to run during compression.

        Returns:
            The compressed documents.
        """

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Async compress retrieved documents given the query context.

        Args:
            documents: The retrieved documents.
            query: The query context.
            callbacks: Optional callbacks to run during compression.

        Returns:
            The compressed documents.
        """
        return await run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )

    def __or__(self, other: BaseDocumentCompressor) -> BaseDocumentCompressor:
        return DocumentCompressorSequence(self, other)


class DocumentCompressorSequence(BaseDocumentCompressor):
    """Sequence of Compressors, which will be executed in order.

    A DocumentCompressorSequence can be instantiated directly or more commonly by using
    the `|` operator where both the left and right operands be a BaseDocumentCompressor.
    """

    _compressor_list: Sequence[BaseDocumentCompressor] = []
    """The Sequence of Compressors to execute."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = "forbid"

    def __init__(
        self,
        first_compressor: BaseDocumentCompressor,
        second_compressor: BaseDocumentCompressor,
        *compressor_list: BaseDocumentCompressor,
    ):
        """Create a new DocumentCompressorSequence.

        Args:
            first_compressor: The first BaseDocumentCompressor in the sequence.
            second_compressor: The second BaseDocumentCompressor in the sequence.
            compressor_list: The rest BaseDocumentCompressor in the sequence.

        first_compressor and second_compressor are place holder to make sure
        that the sequence has least two compressors.
        """
        super().__init__()
        self._compressor_list = [first_compressor, second_compressor, *compressor_list]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using the compressors in the chain.

        Args:
            documents: The retrieved documents.
            query: The query context.
            callbacks: Optional callbacks to run during compression.

        Returns:
            The compressed documents.
        """

        for compressor in self._compressor_list:
            # avoid meaningless compression
            if len(documents) == 0:
                break
            documents = compressor.compress_documents(
                documents, query, callbacks=callbacks
            )
        return documents
