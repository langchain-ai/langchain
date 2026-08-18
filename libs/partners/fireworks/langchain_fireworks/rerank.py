"""Fireworks document reranking integration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import secret_from_env
from openai import OpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import override


class FireworksRerank(BaseDocumentCompressor):
    """Document compressor that uses Fireworks' reranking API."""

    client: Any = None
    """OpenAI-compatible client used to call Fireworks."""

    top_n: int | None = 3
    """Number of documents to return."""

    model: str
    """Fireworks reranking model to use."""

    fireworks_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("FIREWORKS_API_KEY", default=None)
    )
    """Fireworks API key."""

    base_url: str = "https://api.fireworks.ai/inference/v1"
    """Base URL for the Fireworks API."""

    user_agent: str = "langchain:partner"
    """Identifier for the application making the request."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def validate_environment(self) -> FireworksRerank:
        """Create the OpenAI-compatible client when one was not supplied."""
        if self.client is None:
            api_key = (
                self.fireworks_api_key.get_secret_value()
                if self.fireworks_api_key is not None
                else None
            )
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                default_headers={"User-Agent": self.user_agent},
            )
        return self

    def _document_to_str(
        self,
        document: str | Document | Mapping[str, Any],
        rank_fields: Sequence[str] | None = None,
    ) -> str:
        """Convert a supported document value to the string API format."""
        if isinstance(document, Document):
            return document.page_content
        if isinstance(document, Mapping):
            value: Mapping[str, Any] = document
            if rank_fields is not None:
                value = {key: document[key] for key in rank_fields if key in document}
            return json.dumps(value, ensure_ascii=False, default=str)
        return document

    def rerank(
        self,
        documents: Sequence[str | Document | Mapping[str, Any]],
        query: str,
        *,
        rank_fields: Sequence[str] | None = None,
        model: str | None = None,
        top_n: int | None = -1,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return document indexes ordered by relevance to a query.

        Args:
            documents: Documents to rerank.
            query: Query used for reranking.
            rank_fields: Mapping fields to include when serializing mappings.
            model: Model to use instead of the configured model.
            top_n: Number of results to return. `None` returns all results.
            task: Optional task instruction for the reranking model.

        Returns:
            Reranking results containing each document index and score.
        """
        if not documents:
            return []

        requested_top_n = top_n if top_n is None or top_n > 0 else self.top_n
        payload: dict[str, Any] = {
            "model": model or self.model,
            "query": query,
            "documents": [
                self._document_to_str(document, rank_fields) for document in documents
            ],
            "return_documents": False,
        }
        if requested_top_n is not None:
            payload["top_n"] = requested_top_n
        if task is not None:
            payload["task"] = task

        response = self.client.post("/rerank", cast_to=dict, body=payload)
        return [
            {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            for result in response["data"]
        ]

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress documents by keeping the most relevant results."""
        compressed = []
        for result in self.rerank(documents, query):
            document = documents[result["index"]]
            document_copy = Document(
                document.page_content,
                metadata=deepcopy(document.metadata),
            )
            document_copy.metadata["relevance_score"] = result["relevance_score"]
            compressed.append(document_copy)
        return compressed
