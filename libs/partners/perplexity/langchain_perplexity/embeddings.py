"""Wrapper around Perplexity Embeddings API."""

from __future__ import annotations

import base64
import struct
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from perplexity import AsyncPerplexity, Perplexity
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


def _decode_int8_embedding(b64: str) -> list[float]:
    """Decode a `base64_int8`-encoded Perplexity embedding into a list of floats."""
    raw = base64.b64decode(b64)
    return [float(v) for v in struct.unpack(f"<{len(raw)}b", raw)]


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
        embeddings = PerplexityEmbeddings(model="pplx-embed-v1-0.6b")
        ```

    !!! note
        Perplexity returns base64-encoded signed int8 embeddings. This class
        decodes them into `list[float]` values in the range [-128, 127]. The
        magnitude is preserved from the API's quantized output; cosine
        similarity is unaffected by the lack of unit-length normalization.
    """

    client: Any = Field(default=None, exclude=True)
    """Perplexity SDK client (set automatically)."""

    async_client: Any = Field(default=None, exclude=True)
    """Async Perplexity SDK client (set automatically)."""

    model: str = "pplx-embed-v1-4b"
    """Name of the Perplexity embedding model to use.

    See the API reference for available identifiers, including
    `pplx-embed-v1-0.6b` and `pplx-embed-v1-4b`. Contextualized variants are
    served through a separate endpoint and are not exposed by this class.
    """

    pplx_api_key: SecretStr | None = Field(
        default_factory=secret_from_env(
            ["PPLX_API_KEY", "PERPLEXITY_API_KEY"], default=None
        ),
        alias="api_key",
    )
    """Perplexity API key. Reads from `PPLX_API_KEY` or `PERPLEXITY_API_KEY`."""

    request_timeout: float | tuple[float, float] | None = Field(None, alias="timeout")
    """Timeout for requests to the Perplexity embeddings API."""

    max_retries: int = 6
    """Maximum number of retries to make when calling the embeddings API."""

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
        client_params: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": self.max_retries,
        }
        if self.request_timeout is not None:
            client_params["timeout"] = self.request_timeout

        if self.client is None:
            self.client = Perplexity(**client_params)
        if self.async_client is None:
            self.async_client = AsyncPerplexity(**client_params)
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
        return [_decode_int8_embedding(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string using the Perplexity embeddings API.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector for the input text.
        """
        return self.embed_documents([text])[0]

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
        return [_decode_int8_embedding(item.embedding) for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed a single query string.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector for the input text.
        """
        result = await self.aembed_documents([text])
        return result[0]
