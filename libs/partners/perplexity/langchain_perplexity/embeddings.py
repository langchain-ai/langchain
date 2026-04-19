"""Perplexity AI embeddings."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from perplexity import AsyncPerplexity, Perplexity
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class PerplexityEmbeddings(BaseModel, Embeddings):
    """`Perplexity AI` embeddings model integration.

    Setup:
        Install `langchain-perplexity` and set the environment variable
        `PPLX_API_KEY`.

        ```bash
        pip install -U langchain-perplexity
        export PPLX_API_KEY="your-api-key"
        ```

    Key init args:
        model:
            Name of the Perplexity embedding model to use.
        pplx_api_key:
            Perplexity API key. If not provided, read from `PPLX_API_KEY`
            or `PERPLEXITY_API_KEY` environment variables.

    Instantiate:

        ```python
        from langchain_perplexity import PerplexityEmbeddings

        embeddings = PerplexityEmbeddings(model="pplx-embed-v1")
        ```

    Embed a single query:

        ```python
        vector = embeddings.embed_query("what is perplexity AI?")
        ```

    Embed documents:

        ```python
        vectors = embeddings.embed_documents([
            "Perplexity AI builds LLM-powered products.",
            "RAG combines retrieval with generation.",
        ])
        ```
    """

    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)

    model: str = "pplx-embed-v1"
    """Embedding model name."""

    pplx_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("PPLX_API_KEY", default=None), alias="api_key"
    )
    """Perplexity API key."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"pplx_api_key": "PPLX_API_KEY"}

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate API key and initialize clients."""
        api_key = self.pplx_api_key.get_secret_value() if self.pplx_api_key else None
        if not self.client:
            self.client = Perplexity(api_key=api_key)
        if not self.async_client:
            self.async_client = AsyncPerplexity(api_key=api_key)
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one per input text.
        """
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector for the input text.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed a list of documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one per input text.
        """
        response = await self.async_client.embeddings.create(
            input=texts, model=self.model
        )
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed a single query string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector for the input text.
        """
        return (await self.aembed_documents([text]))[0]
