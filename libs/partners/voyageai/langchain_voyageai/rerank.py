from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union

import voyageai  # type: ignore
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.utils import convert_to_secret_str
from pydantic import ConfigDict, SecretStr, model_validator
from voyageai.object import RerankingObject  # type: ignore


class VoyageAIRerank(BaseDocumentCompressor):
    """Document compressor that uses `VoyageAI Rerank API`."""

    client: voyageai.Client = None  # type: ignore
    aclient: voyageai.AsyncClient = None  # type: ignore
    """VoyageAI clients to use for compressing documents."""
    voyage_api_key: Optional[SecretStr] = None
    """VoyageAI API key. Must be specified directly or via environment variable 
        VOYAGE_API_KEY."""
    model: str
    """Model to use for reranking."""
    top_k: Optional[int] = None
    """Number of documents to return."""
    truncation: bool = True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        voyage_api_key = values.get("voyage_api_key") or os.getenv(
            "VOYAGE_API_KEY", None
        )
        if voyage_api_key:
            api_key_secretstr = convert_to_secret_str(voyage_api_key)
            values["voyage_api_key"] = api_key_secretstr

            api_key_str = api_key_secretstr.get_secret_value()
        else:
            api_key_str = None

        values["client"] = voyageai.Client(api_key=api_key_str)
        values["aclient"] = voyageai.AsyncClient(api_key=api_key_str)

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
