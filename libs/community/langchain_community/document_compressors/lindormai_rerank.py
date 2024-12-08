"""Lindorm AI rerank model."""

from __future__ import annotations

import logging
from concurrent.futures import Executor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import run_in_executor
from pydantic import Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

HTTP_HDR_AK_KEY = "x-ld-ak"
HTTP_HDR_SK_KEY = "x-ld-sk"
REST_URL_PATH = "/v1/ai"
REST_URL_MODELS_PATH = REST_URL_PATH + "/models"
INFER_INPUT_KEY = "input"
INFER_PARAMS_KEY = "params"
RSP_DATA_KEY = "data"
RSP_MODELS_KEY = "models"


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
            if values.get("username") is None:
                raise ValueError("username can't be empty")
            if values.get("password") is None:
                raise ValueError("password can't be empty")
            if values.get("endpoint") is None:
                raise ValueError("endpoint can't be empty")
        if not values.get("executor"):
            max_workers = values.get("max_workers", 5)
            values["executor"] = ThreadPoolExecutor(max_workers=max_workers)
        return values

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4))
    def __post_with_retry(
        self, url: str, data: Any = None, json: Any = None, **kwargs: Any
    ) -> Any:
        response = requests.post(url=url, data=data, json=json, **kwargs)
        response.raise_for_status()
        return response

    def _infer(self, model_name: str, input_data: Any, params: Any) -> Any:
        url = f"{self.endpoint}{REST_URL_MODELS_PATH}/{model_name}/infer"
        infer_dict = {INFER_INPUT_KEY: input_data, INFER_PARAMS_KEY: params}
        result = None
        try:
            headers = {HTTP_HDR_AK_KEY: self.username, HTTP_HDR_SK_KEY: self.password}
            response = self.__post_with_retry(url, json=infer_dict, headers=headers)
            response.raise_for_status()
            result = response.json()
        except Exception as error:
            logger.error(
                f"infer model for {model_name} with "
                f"input {input_data} and params {params}: {error}"
            )
        return result[RSP_DATA_KEY] if result else None

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
        response = self._infer(
            model_name=self.model_name,
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
