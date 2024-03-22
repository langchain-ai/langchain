from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Sequence, Union

import voyageai  # type: ignore
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from voyageai.object import RerankingObject  # type: ignore


class VoyageAIRerank(BaseDocumentCompressor):
    """Document compressor that uses `VoyageAI Rerank API`."""

    client: voyageai.Client = None
    aclient: voyageai.AsyncClient = None
    """VoyageAI clients to use for compressing documents."""
    voyageai_api_key: str
    """VoyageAI API key. Must be specified directly or via environment variable 
        VOYAGE_API_KEY."""
    model: str
    """Model to use for reranking."""
    top_k: Optional[int] = None
    """Number of documents to return."""
    truncation: bool = True

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        voyageai_api_key = get_from_dict_or_env(
            values, "voyageai_api_key", "VOYAGE_API_KEY"
        )
        values["client"] = voyageai.Client(api_key=voyageai_api_key)
        values["aclient"] = voyageai.AsyncClient(api_key=voyageai_api_key)

        return values

    def _rerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
    ) -> RerankingObject:
        """Returns an ordered list of documents ordered by their relevance
        to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
        """
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        return self.client.rerank(
            query=query,
            documents=docs,
            model=self.model,
            top_k=self.top_k,
            truncation=self.truncation,
        )

    async def _arerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
    ) -> RerankingObject:
        """Returns an ordered list of documents ordered by their relevance
        to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
        """
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        return await self.aclient.rerank(
            query=query,
            documents=docs,
            model=self.model,
            top_k=self.top_k,
            truncation=self.truncation,
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using VoyageAI's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents in relevance_score order.
        """
        if len(documents) == 0:
            return []

        compressed = []
        for res in self._rerank(documents, query).results:
            doc = documents[res.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res.relevance_score
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using VoyageAI's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents in relevance_score order.
        """
        if len(documents) == 0:
            return []

        compressed = []
        for res in (await self._arerank(documents, query)).results:
            doc = documents[res.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res.relevance_score
            compressed.append(doc_copy)
        return compressed
