from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from perplexity import AsyncPerplexity
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langchain_perplexity._utils import initialize_client


class PerplexityEmbeddings(BaseModel, Embeddings):
    """Perplexity embedding model.

    To use, you should have the ``perplexity`` python package installed, and the
    environment variable ``PPLX_API_KEY`` (or ``PERPLEXITY_API_KEY``) set with your
    API key, or pass it as the ``pplx_api_key`` parameter.

    See the Perplexity Embeddings documentation:
    - https://docs.perplexity.ai/docs/embeddings
    - https://docs.perplexity.ai/api-reference/embeddings-post

    Example:
        Basic usage:

        .. code-block:: python

            from langchain_perplexity import PerplexityEmbeddings

            embeddings = PerplexityEmbeddings()

            query_vector = embeddings.embed_query("hello world")
            doc_vectors = embeddings.embed_documents(["hello", "world"])

        Selecting a specific model:

        .. code-block:: python

            from langchain_perplexity import PerplexityEmbeddings

            embeddings = PerplexityEmbeddings(model="pplx-embed-v1-4b")
            vectors = embeddings.embed_documents(
                ["The quick brown fox", "jumps over the lazy dog"]
            )
    """

    model: str = "pplx-embed-v1-4b"
    """Name of the Perplexity embedding model to use."""

    client: Any = Field(default=None, exclude=True)
    """Perplexity SDK client (set automatically)."""

    async_client: Any = Field(default=None, exclude=True)
    """Async Perplexity SDK client (set automatically)."""

    pplx_api_key: SecretStr = Field(default=SecretStr(""))
    """Perplexity API key. Falls back to ``PPLX_API_KEY`` or ``PERPLEXITY_API_KEY``."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and initialize the Perplexity client."""
        values = initialize_client(values)
        if not values.get("async_client"):
            api_key = (
                values["pplx_api_key"].get_secret_value()
                if values.get("pplx_api_key")
                else None
            )
            values["async_client"] = AsyncPerplexity(api_key=api_key)
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using the Perplexity embeddings API.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, one per input text.
        """
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [r.embedding for r in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string using the Perplexity embeddings API.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector for the input text.
        """
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed a list of documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, one per input text.
        """
        response = await self.async_client.embeddings.create(
            model=self.model, input=texts
        )
        return [r.embedding for r in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed a single query string.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector for the input text.
        """
        response = await self.async_client.embeddings.create(
            model=self.model, input=[text]
        )
        return response.data[0].embedding
