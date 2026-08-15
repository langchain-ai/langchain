"""Fireworks Reranker for document compression."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import secret_from_env
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class FireworksRerank(BaseModel):
    """Document reranker powered by Fireworks.

    This reranker uses the Fireworks API to reorder documents based on their
    relevance to a query. It implements the `BaseDocumentCompressor` interface
    for use with LangChain's contextual compression retrievers.

    Setup:
        Install `langchain_fireworks` and set environment variable
        `FIREWORKS_API_KEY`.

        ```bash
        pip install -U langchain_fireworks
        export FIREWORKS_API_KEY="your-api-key"
        ```

    Key init args:
        model: Name of Fireworks reranking model to use.
            Default: "accounts/fireworks/models/reranker"
        top_n: Number of documents to return. Default: 3
        fireworks_api_key: Fireworks API key.

    Example:
        ```python
        from langchain_fireworks import FireworksRerank
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

        reranker = FireworksRerank(top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )
        compressed_docs = compression_retriever.invoke("your query")
        ```
    """

    client: Any = Field(default=None, exclude=True)  # type: ignore[assignment]
    """Fireworks API client for reranking."""

    top_n: int | None = 3
    """Number of documents to return."""

    model: str = "accounts/fireworks/models/reranker"
    """Model to use for reranking."""

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

    Automatically read from env variable `FIREWORKS_API_KEY` if not provided.
    """

    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment variables and initialize client."""
        self.client = OpenAI(
            api_key=self.fireworks_api_key.get_secret_value(),
            base_url="https://api.fireworks.ai/inference/v1",
        )
        return self

    def rerank(
        self,
        documents: Sequence[str | Document | dict],
        query: str,
        *,
        model: str | None = None,
        top_n: int | None = -1,
    ) -> list[dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n: The number of results to return. If `None` returns all results.

        Returns:
            List of dicts containing index and relevance_score.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc
            for doc in documents
        ]

        model = model or self.model
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n

        response = self.client.post(
            "/rerank",
            json={
                "query": query,
                "documents": docs,
                "model": model,
                "top_n": top_n,
            },
        )

        if not response.is_success:
            response.raise_for_status()

        data = response.json()

        if "data" not in data:
            raise ValueError(f"Unexpected response format: {data}")

        return [
            {"index": item["index"], "relevance_score": item["score"]}
            for item in data["data"]
        ]

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
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(
                doc.page_content, metadata=deepcopy(doc.metadata)
            )
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Async compress documents using Fireworks' rerank API."""
        # For now, just call the sync version
        # Can be optimized later with async client
        return self.compress_documents(documents, query, callbacks)