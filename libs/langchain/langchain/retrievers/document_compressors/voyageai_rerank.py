from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import voyageai
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.utils import get_from_dict_or_env


class VoyageAIRerank(BaseDocumentCompressor):
    """Document compressor that uses `VoyageAI Rerank API`."""

    _client: voyageai.Client = None
    _aclient: voyageai.AsyncClient = None
    """VoyageAI clients to use for compressing documents."""
    top_k: Optional[int]
    """Number of documents to return."""
    model: str
    """Model to use for reranking."""
    voyageai_api_key: str
    """VoyageAI API key. Must be specified directly or via environment variable 
        VOYAGE_API_KEY."""
    truncation: bool = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        voyageai_api_key = get_from_dict_or_env(
            values, "voyageai_api_key", "VOYAGE_API_KEY"
        )
        values["_client"] = voyageai.Client(api_key=voyageai_api_key)
        values["_aclient"] = voyageai.AsyncClient(api_key=voyageai_api_key)

        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_k : The number of results to return. If None returns all results.
                Defaults to self.top_k.
        """  # noqa: E501
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_k = top_k if (top_k is None or top_k > 0) else self.top_k
        results = self._client.rerank(
            query=query,
            documents=docs,
            model=model,
            top_k=top_k,
            truncation=self.truncation,
        )
        result_dicts = []
        for res in results.results:
            result_dicts.append(
                {"index": res.index, "relevance_score": res.relevance_score}
            )
        return result_dicts

    async def arerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_k : The number of results to return. If None returns all results.
                Defaults to self.top_k.
        """  # noqa: E501
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_k = top_k if (top_k is None or top_k > 0) else self.top_k
        results = await self._aclient.rerank(
            query=query,
            documents=docs,
            model=model,
            top_k=top_k,
            truncation=self.truncation,
        )
        result_dicts = []
        for res in results.results:
            result_dicts.append(
                {"index": res.index, "relevance_score": res.relevance_score}
            )
        return result_dicts

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
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
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
        compressed = []
        for res in await self.arerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
