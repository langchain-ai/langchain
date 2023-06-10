"""Retriever that wraps a base retriever and filters the results."""

from typing import Any, List, Optional

from pydantic import BaseModel, Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.schema import BaseRetriever, Document


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """Retriever that wraps a base retriever and compresses the results."""

    base_compressor: BaseDocumentCompressor
    """Compressor for compressing retrieved documents."""

    base_retriever: BaseRetriever
    """Base Retriever to use for getting relevant documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        run_manager_ = run_manager or CallbackManagerForRetrieverRun.get_noop_manager()
        docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager_.get_child(), **kwargs
        )
        compressed_docs = self.base_compressor.compress_documents(docs, query)
        return list(compressed_docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        run_manager_ = (
            run_manager or AsyncCallbackManagerForRetrieverRun.get_noop_manager()
        )
        docs = await self.base_retriever.aget_relevant_documents(
            query, callbacks=run_manager_.get_child(), **kwargs
        )
        compressed_docs = await self.base_compressor.acompress_documents(docs, query)
        return list(compressed_docs)
