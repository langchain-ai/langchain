"""Lindorm AI rerank model."""

from __future__ import annotations

import logging
from concurrent.futures import Executor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from importlib import util
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import run_in_executor
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)


class LindormAIRerank(BaseDocumentCompressor):
    """Document compressor that uses `Lindorm AI Rerank API`.

    To use, you should have the ``lindormai`` python package installed,
    and set variable ``endpoint``, ``username``, ``password`` and ``model_name``.

    Example:
    .. code-block:: python
    from langchain_community.document_compressors.lindormai_rerank import
    LindormAIRerank
    lindorm_ai_rerank = LindormAIRerank(
        endpoint='https://ld-xxx-proxy-ml.lindorm.rds.aliyuncs.com:9002',
        username='root',
        password='xxx',
        model_name='rerank_bge_large'
    )
    """

    endpoint: str = Field(
        ...,
        description="The endpoint of Lindorm AI to use.",
    )
    username: str = Field(
        ...,
        description="Lindorm username.",
    )
    password: str = Field(
        ...,
        description="Lindorm password.",
    )
    model_name: str = Field(
        ...,
        description="The model to use for reranking.",
    )
    client: Any = None
    executor: Optional[Executor] = None
    max_workers: Optional[int] = 5
    """Lindorm AI client to use for reranking documents."""
    top_n: Optional[int] = 3
    batch_size: Optional[int] = 32
    """Number of documents to return."""

    class Config:
        """Configuration for this pydantic object."""

        protected_namespaces = ()
        arbitrary_types_allowed = True
        populate_by_name = True

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Ensure the client is initialized properly."""
        if not values.get("client"):
            if util.find_spec("lindormai") is None:
                raise ImportError(
                    "Could not import lindormai python package. "
                    "Please install it with "
                    "`pip install lindormai-x.y-z-py3-none-any.whl`."
                )
            else:
                from lindormai.model_manager import ModelManager

                values["client"] = ModelManager(
                    values["endpoint"], values["username"], values["password"]
                )

        if not values.get("executor"):
            max_workers = values.get("max_workers", 5)
            values["executor"] = ThreadPoolExecutor(max_workers=max_workers)
        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = -1,
        batch_size: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        """
        Returns an ordered list of documents ordered by their relevance to the provided
        query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
            batch_size : The batch size used for the computation.
                Defaults to self.batch_size(32).
        """
        if len(documents) == 0:  # to avoid empty API call
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        batch_size = (
            batch_size if (batch_size is None or batch_size > 0) else self.batch_size
        )

        # Call the Lindorm AI client for reranking
        response = self.client.infer(
            name=self.model_name,
            input_data={"query": query, "chunks": docs},
            params={"topK": top_n, "batchSize": batch_size},
        )

        result_dicts = []
        for res in response:
            result_dicts.append(
                {"index": res["index"], "relevance_score": res["score"]}
            )

        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Lindorm AI's rerank API.
        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.
        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(
                doc.page_content, id=doc.id, metadata=deepcopy(doc.metadata)
            )
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
        Compress documents using Lindorm AI's rerank API.
        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.
        Returns:
            A sequence of compressed documents.
            :param callbacks:
            :param query:
            :param documents:
            :param self:
        """

        return await run_in_executor(
            self.executor, self.compress_documents, documents, query
        )
