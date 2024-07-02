from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence, Union

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import Field
from mixedbread_ai import RankedDocument  # type: ignore

from langchain_mixedbreadai.client import MixedBreadAIClient


class MixedbreadAIRerank(MixedBreadAIClient, BaseDocumentCompressor):
    """
    Document compressor that uses `Mixedbread AI Rerank API`.

    This class utilizes the Mixedbread AI rerank API to reorder documents based
    on their relevance to a given query. It supports both synchronous and
    asynchronous operations.

    Attributes:
        model (str): Model to use for reranking.
            Defaults to "mixedbread-ai/mxbai-rerank-large-v1".
        top_n (int): Number of documents to return. Defaults to 10.
    """

    model: str = Field(
        default="default",
        description="Model to use for reranking.",
        min_length=1,
    )
    top_n: int = Field(default=10, description="Number of documents to return.", ge=1)

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        rank_fields: Optional[Sequence[str]] = None,
    ) -> Sequence[RankedDocument]:
        """
        Rerank documents based on their relevance to the provided query
        using Mixedbread AI's rerank API.

        Args:
            documents (Sequence[Union[str, Document, dict]]):
                A sequence of documents to rerank.
            query (str): The query to use for reranking.
            rank_fields (Optional[Sequence[str]]): Fields to consider for ranking.

        Returns:
            Sequence[RankedDocument]: A sequence of ranked documents.
        """
        if not documents:
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]

        return self._client.reranking(
            model=self.model,
            query=query,
            input=docs,
            rank_fields=rank_fields,
            top_k=self.top_n,
            return_input=True,
            request_options=self._request_options,
        ).data

    async def arerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        rank_fields: Optional[Sequence[str]] = None,
    ) -> Sequence[RankedDocument]:
        """
        Asynchronously rerank documents based on their relevance
        to the provided query using Mixedbread AI's rerank API.

        Args:
            documents (Sequence[Union[str, Document, dict]]):
                A sequence of documents to rerank.
            query (str): The query to use for reranking.
            rank_fields (Optional[Sequence[str]]):
                Fields to consider for ranking.

        Returns:
            Sequence[RankedDocument]: A sequence of ranked documents.
        """
        if not documents:
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]

        return (
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

    def _compress(
        self,
        rerank_results: Sequence[RankedDocument],
        documents: Sequence[Document],
    ) -> Sequence[Document]:
        compressed = []
        for res in rerank_results:
            doc = documents[res.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res.score
            compressed.append(doc_copy)
        return compressed

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
            callbacks (Optional[Callbacks]):
                Callbacks to run during the compression process.

        Returns:
            Sequence[Document]:
                A sequence of compressed documents in relevance_score order.
        """
        rerank_results = self.rerank(documents, query)
        return self._compress(rerank_results, documents)

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
            callbacks (Optional[Callbacks]):
                Callbacks to run during the compression process.

        Returns:
            Sequence[Document]:
                A sequence of compressed documents in relevance_score order.
        """
        rerank_results = await self.arerank(documents, query)
        return self._compress(rerank_results, documents)
