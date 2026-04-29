"""Wrapper around Perplexity Embeddings API."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from perplexity import AsyncPerplexity, Perplexity
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class PerplexityEmbeddings(BaseModel, Embeddings):
    """`Perplexity AI` embeddings.

    Setup:
        Install the `perplexityai` package and set the `PPLX_API_KEY`
        (or `PERPLEXITY_API_KEY`) environment variable, or pass the key as
        the `pplx_api_key`/`api_key` argument.

        ```bash
        pip install -U langchain-perplexity
        export PPLX_API_KEY=your_api_key
        ```

        See the Perplexity Embeddings API reference:
        https://docs.perplexity.ai/api-reference/embeddings-post

        Instantiate:

        ```python
        from langchain_perplexity import PerplexityEmbeddings

        embeddings = PerplexityEmbeddings()
        ```

        Embed a single query:

        ```python
        query_vector = embeddings.embed_query("hello world")
        ```

        Embed documents:

        ```python
        doc_vectors = embeddings.embed_documents(["hello", "world"])
        ```

        Select a specific model:

        ```python
        embeddings = PerplexityEmbeddings(model="pplx-embed-v1-4b")
        ```
    """

    client: Any = Field(default=None, exclude=True)
    """Perplexity SDK client (set automatically)."""

    async_client: Any = Field(default=None, exclude=True)
    """Async Perplexity SDK client (set automatically)."""

    model: str = "pplx-embed-v1-4b"
    """Name of the Perplexity embedding model to use.

    See the API reference for available identifiers, including contextualized
    variants such as `pplx-embed-v1-0.6b` and `pplx-embed-context-v1-4b`.
    """

    pplx_api_key: SecretStr | None = Field(
        default_factory=secret_from_env(
            ["PPLX_API_KEY", "PERPLEXITY_API_KEY"], default=None
        ),
        alias="api_key",
    )
    """Perplexity API key. Reads from `PPLX_API_KEY` or `PERPLEXITY_API_KEY`."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map secret field names to their environment variable names."""
        return {"pplx_api_key": "PPLX_API_KEY"}

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Perplexity SDK clients."""
        if not self.pplx_api_key:
            msg = (
                "Perplexity API key not provided. Pass `pplx_api_key` (or "
                "`api_key`) to PerplexityEmbeddings, or set the `PPLX_API_KEY` "
                "or `PERPLEXITY_API_KEY` environment variable."
            )
            raise ValueError(msg)

        api_key = self.pplx_api_key.get_secret_value()
        if self.client is None:
            self.client = Perplexity(api_key=api_key)
        if self.async_client is None:
            self.async_client = AsyncPerplexity(api_key=api_key)
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using the Perplexity embeddings API.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, one per input text. An empty list is returned
            when `texts` is empty.
        """
        if not texts:
            return []
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
            A list of embeddings, one per input text. An empty list is returned
            when `texts` is empty.
        """
        if not texts:
            return []
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
