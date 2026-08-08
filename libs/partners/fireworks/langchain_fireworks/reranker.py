"""Fireworks reranker wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import requests
from aiohttp import ClientSession, ClientTimeout
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self, override


class FireworksRerank(BaseDocumentCompressor):
    """Document compressor that uses the `Fireworks Rerank API`.

    Setup:

        Install ``langchain_fireworks`` and set environment variable
        ``FIREWORKS_API_KEY``.

        ```bash
        pip install -U langchain_fireworks
        export FIREWORKS_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of Fireworks reranker model to use.

    Key init args — client params:
        fireworks_api_key:
            Fireworks API key.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:

        ```python
        from langchain_fireworks import FireworksRerank

        compressor = FireworksRerank(
            model="fireworks/qwen3-reranker-8b",
        )
        ```

    Rerank documents:

        ```python
        from langchain_core.documents import Document

        docs = [
            Document("Paris is the capital of France."),
            Document("France is a country in Western Europe."),
        ]
        results = compressor.compress_documents(docs, "What is the capital of France?")
        ```
    """

    fireworks_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "FIREWORKS_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `FIREWORKS_API_KEY`."
            ),
        ),
    )
    """Fireworks API key.

    Automatically read from env variable ``FIREWORKS_API_KEY`` if not provided.
    """

    fireworks_api_base: str = "https://api.fireworks.ai/inference/v1"
    """Base URL for the Fireworks inference API."""

    model: str = "fireworks/qwen3-reranker-8b"
    """Model to use for reranking."""

    top_n: int | None = None
    """Number of documents to return. If ``None``, returns all documents."""

    return_documents: bool = True
    """Whether to return the document text in the API response."""

    task: str | None = None
    """Optional task description to guide the reranking process."""

    timeout: int = 30
    """Timeout in seconds for requests to the Fireworks API."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def _validate_environment(self) -> Self:
        """Validate that the API key is available."""
        self.fireworks_api_key.get_secret_value()
        return self

    def _build_payload(
        self,
        documents: Sequence[str],
        query: str,
        *,
        model: str | None = None,
        top_n: int | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON payload for the rerank request."""
        payload: dict[str, Any] = {
            "model": model if model is not None else self.model,
            "query": query,
            "documents": list(documents),
            "return_documents": self.return_documents,
        }
        effective_top_n = top_n if top_n is not None else self.top_n
        if effective_top_n is not None:
            payload["top_n"] = effective_top_n
        effective_task = task if task is not None else self.task
        if effective_task is not None:
            payload["task"] = effective_task
        return payload

    def rerank(
        self,
        documents: Sequence[str | Document],
        query: str,
        *,
        model: str | None = None,
        top_n: int | None = None,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns an ordered list of documents by their relevance to the query.

        Args:
            documents: A sequence of documents to rerank.
            query: The query to use for reranking.
            model: Model to use for reranking. Defaults to ``self.model``.
            top_n: Number of results to return. Defaults to ``self.top_n``.
            task: Task description to guide the reranking process.

        Returns:
            List of dicts with ``index`` and ``relevance_score`` keys,
            ordered by relevance (highest first).
        """
        if len(documents) == 0:
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        headers = {
            "Authorization": f"Bearer {self.fireworks_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(docs, query, model=model, top_n=top_n, task=task)
        response = requests.post(
            url=f"{self.fireworks_api_base}/rerank",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code >= 500:
            msg = f"Fireworks Server: Error {response.status_code}"
            raise Exception(msg)
        if response.status_code >= 400:
            msg = f"Fireworks received an invalid payload: {response.text}"
            raise ValueError(msg)
        if response.status_code != 200:
            msg = (
                f"Fireworks returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )
            raise Exception(msg)
        data = response.json()
        return [
            {"index": item["index"], "relevance_score": item["relevance_score"]}
            for item in data["data"]
        ]

    async def _arerank(
        self,
        documents: Sequence[str],
        query: str,
        *,
        model: str | None = None,
        top_n: int | None = None,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async version of rerank."""
        headers = {
            "Authorization": f"Bearer {self.fireworks_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(
            documents, query, model=model, top_n=top_n, task=task
        )
        async with (
            ClientSession() as session,
            session.post(
                f"{self.fireworks_api_base}/rerank",
                json=payload,
                headers=headers,
                timeout=ClientTimeout(total=self.timeout),
            ) as response,
        ):
            if response.status >= 500:
                msg = f"Fireworks Server: Error {response.status}"
                raise Exception(msg)
            if response.status >= 400:
                text = await response.text()
                msg = f"Fireworks received an invalid payload: {text}"
                raise ValueError(msg)
            if response.status != 200:
                text = await response.text()
                msg = (
                    f"Fireworks returned an unexpected response with status "
                    f"{response.status}: {text}"
                )
                raise Exception(msg)
            response_json = await response.json()
        return [
            {"index": item["index"], "relevance_score": item["relevance_score"]}
            for item in response_json["data"]
        ]

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress documents using Fireworks' rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents ordered by relevance.
        """
        if len(documents) == 0:
            return []
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

    @override
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Async compress documents using Fireworks' rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents ordered by relevance.
        """
        if len(documents) == 0:
            return []
        docs = [doc.page_content for doc in documents]
        results = await self._arerank(docs, query)
        compressed = []
        for res in results:
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
