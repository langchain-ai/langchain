from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Sequence, Union

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from .client import MixedBreadAIClient


class MixedbreadAIRerank(MixedBreadAIClient, BaseDocumentCompressor):
    """Document compressor that uses `Mixedbread AI Rerank API`.

    This class utilizes the Mixedbread AI Rerank API to reorder documents based
    on their relevance to a given query. It supports both synchronous and
    asynchronous operations.

    Attributes:
        model (str): Model to use for reranking.
                Defaults to "mixedbread-ai/mxbai-rerank-large-v1".
        top_n (int): Number of documents to return. Defaults to 3.
    """

    model: str = "mixedbread-ai/mxbai-rerank-large-v1"
    """Model to use for reranking."""
    top_n: int = 10
    """Number of documents to return."""

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        rank_fields: Optional[Sequence[str]] = None,
    ) -> Sequence[Dict]:
        """
        Rerank documents based on their relevance to the provided query.

        Args:
            documents (Sequence[Union[str, Document, dict]]): A sequence of documents
                to rerank.
            query (str): The query to use for reranking.
            rank_fields (Optional[Sequence[str]]): Fields to consider for ranking.

        Returns:
            Sequence[Dict]: A sequence of dictionaries containing document index
                and score.
        """
        if len(documents) == 0:
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]

        results = self._client.reranking(
            model=self.model,
            query=query,
            input=docs,
            rank_fields=rank_fields,
            top_k=self.top_n,
            return_input=False,
            request_options=self._request_options,
        ).data

        return [{"index": result.index, "score": result.score} for result in results]

    async def arerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        rank_fields: Optional[Sequence[str]] = None,
    ) -> Sequence[Dict]:
        """
        Asynchronously rerank documents based on their relevance to the provided query.

        Args:
            documents (Sequence[Union[str, Document, dict]]): A sequence of documents
                to rerank.
            query (str): The query to use for reranking.
            rank_fields (Optional[Sequence[str]]): Fields to consider for ranking.

        Returns:
            Sequence[Dict]: A sequence of dictionaries containing document index
                and score.
        """
        if len(documents) == 0:
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]

        results = (
            await self._aclient.reranking(
                model=self.model,
                query=query,
                input=docs,
                rank_fields=rank_fields,
                top_k=self.top_n,
                return_input=False,
                request_options=self._request_options,
            )
        ).data

        return [{"index": result.index, "score": result.score} for result in results]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Mixedbread AI's rerank API.

        Args:
            documents (Sequence[Document]): A sequence of documents to compress.
            query (str): The query to use for compressing the documents.
            callbacks (Optional[Callbacks]): Callbacks to run during the compression
                process.

        Returns:
            Sequence[Document]: A sequence of compressed documents in relevance_score
                order.
        """
        if len(documents) == 0:
            return []

        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["score"]
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Mixedbread AI's rerank API asynchronously.

        Args:
            documents (Sequence[Document]): A sequence of documents to compress.
            query (str): The query to use for compressing the documents.
            callbacks (Optional[Callbacks]): Callbacks to run during the compression
                process.

        Returns:
            Sequence[Document]: A sequence of compressed documents in relevance_score
                order.
        """
        if len(documents) == 0:
            return []

        compressed = []
        for res in await self.arerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["score"]
            compressed.append(doc_copy)
        return compressed
