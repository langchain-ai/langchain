"""OpenRouter embeddings models."""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterEmbeddings(BaseModel, Embeddings):
    """OpenRouter embedding model integration.

    Setup:
        Install `langchain-openrouter` and set environment variable
        `OPENROUTER_API_KEY`.

        ```bash
        pip install -U langchain-openrouter
        export OPENROUTER_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of OpenRouter model to use, e.g.
            `"qwen/qwen3-embedding-8b"`.
        dimensions:
            The number of dimensions the resulting output embeddings
            should have.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            OpenRouter API key. If not passed in will be read from env var
            `OPENROUTER_API_KEY`.

    See full list of supported init args and their descriptions in the
    params section.

    Instantiate:
        ```python
        from langchain_openrouter import OpenRouterEmbeddings

        embeddings = OpenRouterEmbeddings(
            model="qwen/qwen3-embedding-8b",
            # api_key="...",
            # other params...
        )
        ```

    Embed single text:
        ```python
        input_text = "The meaning of life is 42"
        vector = embeddings.embed_query(input_text)
        print(vector[:3])
        ```

        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    Embed multiple texts:
        ```python
        input_texts = ["Document 1...", "Document 2..."]
        vectors = embeddings.embed_documents(input_texts)
        print(len(vectors))
        # The first 3 coordinates for the first vector
        print(vectors[0][:3])
        ```

        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    Async:
        ```python
        vector = await embeddings.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embeddings.aembed_documents(input_texts)
        ```

        ```python
        [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
        ```
    """

    model: str = Field()
    """The name of the model"""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    """OpenRouter API key"""
    api_base: str = Field(
        default_factory=from_env("OPENROUTER_API_BASE", default=DEFAULT_API_BASE),
    )
    """OpenRouter API base URL"""
    dimensions: int | None = None
    """The number of dimensions the resulting output embeddings should have.

    Only supported in text-embedding-3 and later models.
    """

    # Client parameters
    request_timeout: float | None = Field(default=None, alias="timeout")
    """Timeout for requests."""
    max_retries: int = 3
    """Maximum number of retries."""
    default_headers: dict[str, str] | None = None
    """Default headers to include in requests."""
    default_query: dict[str, str] | None = None
    """Default query parameters to include in requests."""
    http_client: Any = None
    """Optional httpx.Client to use for making requests."""
    http_async_client: Any = None
    """Optional httpx.AsyncClient to use for making requests."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of embeddings."""
        return "embeddings-openrouter"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "OPENROUTER_API_KEY"}

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, OPENROUTER_API_KEY must be set."
            raise ValueError(msg)

        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (getattr(self, "_root_client", None) or None):
            sync_specific: dict = {"http_client": self.http_client}
            self._root_client = openai.OpenAI(**client_params, **sync_specific)
            self._client = self._root_client.embeddings
        if not (getattr(self, "_root_async_client", None) or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self._root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self._async_client = self._root_async_client.embeddings
        return self

    def _get_request_payload(self, text_input: str | list[str]) -> dict:
        """Get the request payload for embeddings."""
        text = (
            text_input
            if isinstance(text_input, str)
            else text_input[0]
            if text_input
            else ""
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
        }

        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        return payload

    def _embed_with_payload(self, text: str) -> list[float]:
        """Embed text using request payload."""
        payload = self._get_request_payload(text)
        response = self._client.create(**payload)
        return response.data[0].embedding

    async def _aembed_with_payload(self, text: str) -> list[float]:
        """Embed text using request payload asynchronously."""
        payload = self._get_request_payload(text)
        response = await self._async_client.create(**payload)
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        if not hasattr(self, "_client") or not self._client:
            msg = "OpenRouter client is not initialized."
            raise ValueError(msg)

        # OpenRouter API doesn't support batch embedding, so we need to make
        # individual calls and combine results
        return [self._embed_with_payload(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        if not hasattr(self, "_async_client") or not self._async_client:
            msg = "OpenRouter async client is not initialized."
            raise ValueError(msg)

        # OpenRouter API doesn't support batch embedding, so we need to make
        # individual calls and combine results
        return [await self._aembed_with_payload(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
