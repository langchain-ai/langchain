"""Interface for retrieved document filters."""
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from pydantic import BaseModel

from langchain.document_transformers import DocumentTransformerPipeline
from langchain.schema import BaseDocumentTransformer, Document


class DocumentCompressorMixin(ABC):
    """"""

    @abstractmethod
    def compress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""

    @abstractmethod
    async def acompress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""


class BaseDocumentCompressor(
    DocumentCompressorMixin, BaseDocumentTransformer, BaseModel, ABC
):
    """Interface for retrieved document compressors."""

    def transform_documents(
        self, documents: Sequence[Document], query: Optional[str] = None, **kwargs: Any
    ) -> Sequence[Document]:
        """"""
        if query is None:
            raise ValueError(
                "Keyword argument `query` must be non-null when passed in to "
                "BaseDocumentCompressor.transform_documents."
            )
        return self.compress_documents(documents, query)

    async def atransform_documents(
        self, documents: Sequence[Document], query: Optional[str] = None, **kwargs: Any
    ) -> Sequence[Document]:
        """"""
        if query is None:
            raise ValueError(
                "Keyword argument `query` must be non-null when passed in to "
                "BaseDocumentCompressor.transform_documents."
            )
        return await self.acompress_documents(documents, query)


class DocumentCompressorPipeline(DocumentCompressorMixin, DocumentTransformerPipeline):
    """Document compressor that uses a pipeline of transformers."""

    def compress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        return self.transform_documents(documents, query=query)

    async def acompress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        return await self.atransform_documents(documents, query=query)
