"""OpenRouter embeddings integration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from typing_extensions import Self

logger = logging.getLogger(__name__)

MAX_TOKENS = 16_000
"""A batching parameter for the OpenRouter API. This is NOT the maximum number of
tokens accepted by the embedding model for each document/chunk, but rather the
maximum number of tokens that can be sent in a single request to the OpenRouter
API (across multiple documents/chunks)."""

_RETRY_STATUS_CODE_RATE_LIMIT = 429
_RETRY_STATUS_CODE_SERVER_ERROR = 500


def _is_retryable_error(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry.

    Only retries on:
    - Timeout exceptions
    - 429 (rate limit) errors
    - 5xx (server) errors

    Does NOT retry on 400 (bad request) or other 4xx client errors.
    """
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        return (
            status_code == _RETRY_STATUS_CODE_RATE_LIMIT
            or status_code >= _RETRY_STATUS_CODE_SERVER_ERROR
        )
    return False


class OpenRouterEmbeddings(BaseModel, Embeddings):
    r"""OpenRouter embedding model integration.

    OpenRouter provides unified API access to embedding models from multiple
    providers. See https://openrouter.ai/models?supported_parameters=embeddings
    for available models.

    ???+ info "Setup"

        Install `langchain-openrouter` and set environment variable
        `OPENROUTER_API_KEY`.

        ```bash
        pip install -U langchain-openrouter
        ```

        ```bash
        export OPENROUTER_API_KEY="your-api-key"
        ```

    ??? info "Key init args — completion params"

        | Param | Type | Description |
        | ----- | ---- | ----------- |
        | `model` | `str` | Model name to use for embeddings. |

    ??? info "Key init args — client params"

        | Param | Type | Description |
        | ----- | ---- | ----------- |
        | `api_key` | `str \| None` | OpenRouter API key. |
        | `base_url` | `str \| None` | Base URL for API requests. |
        | `timeout` | `int \| None` | Timeout in seconds. |
        | `max_retries` | `int` | Max retries (default `5`). |
        | `app_url` | `str \| None` | App URL for attribution. |
        | `app_title` | `str \| None` | App title for attribution. |

    ??? info "Instantiate"

        ```python
        from langchain_openrouter import OpenRouterEmbeddings

        embed = OpenRouterEmbeddings(
            model="openai/text-embedding-3-small",
            # api_key="...",
            # other params...
        )
        ```

    See https://openrouter.ai/docs for platform documentation.
    """

    client: httpx.Client = Field(default=None)  # type: ignore[assignment]
    """Underlying synchronous HTTP client."""

    async_client: httpx.AsyncClient = Field(  # type: ignore[assignment]
        default=None
    )

    openrouter_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    """OpenRouter API key."""

    openrouter_api_base: str | None = Field(
        default_factory=from_env("OPENROUTER_API_BASE", default=None),
        alias="base_url",
    )
    """OpenRouter API base URL. Defaults to OpenRouter's public API."""

    app_url: str | None = Field(
        default_factory=from_env(
            "OPENROUTER_APP_URL",
            default="https://docs.langchain.com",
        ),
    )
    """Application URL for OpenRouter attribution.

    Maps to `HTTP-Referer` header.

    Defaults to LangChain docs URL. Set this to your app's URL to get
    attribution for API usage in the OpenRouter dashboard.

    See https://openrouter.ai/docs/app-attribution for details.
    """

    app_title: str | None = Field(
        default_factory=from_env("OPENROUTER_APP_TITLE", default="LangChain"),
    )
    """Application title for OpenRouter attribution.

    Maps to `X-Title` header.

    Defaults to `'LangChain'`. Set this to your app's name to get attribution
    for API usage in the OpenRouter dashboard.

    See https://openrouter.ai/docs/app-attribution for details.
    """

    timeout: int = 120
    """Timeout for requests in seconds."""

    max_retries: int | None = 5
    """Maximum number of retries."""

    wait_time: int | None = 30
    """Wait time in seconds between retries."""

    max_concurrent_requests: int = 64
    """Maximum number of concurrent requests to make to the OpenRouter API."""

    model: str = "openai/text-embedding-3-small"
    """The name of the model to use for embeddings."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate configuration."""
        api_key = (
            self.openrouter_api_key.get_secret_value()
            if self.openrouter_api_key
            else None
        )

        if not api_key:
            msg = "OPENROUTER_API_KEY must be set."
            raise ValueError(msg)

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title

        base_url = self.openrouter_api_base or "https://openrouter.ai/api/v1"

        if not self.client:
            self.client = httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=self.timeout,
            )
        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self

    def _get_batches(self, texts: list[str]) -> Iterable[list[str]]:
        """Split list of texts into batches based on character count.

        Since OpenRouter doesn't provide a tokenizer, we use a simple
        heuristic of ~4 characters per token.
        """
        batch: list[str] = []
        batch_chars = 0

        for text in texts:
            text_chars = len(text)
            text_tokens_estimate = text_chars // 4

            if batch_chars + text_tokens_estimate > MAX_TOKENS:
                if len(batch) > 0:
                    yield batch
                batch = [text]
                batch_chars = text_tokens_estimate
            else:
                batch.append(text)
                batch_chars += text_tokens_estimate
        if batch:
            yield batch

    def _retry(self, func: Callable) -> Callable:
        if self.max_retries is None or self.wait_time is None:
            return func

        return retry(
            retry=retry_if_exception(_is_retryable_error),
            wait=wait_fixed(self.wait_time),
            stop=stop_after_attempt(self.max_retries),
        )(func)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        try:

            @self._retry
            def _embed_batch(batch: list[str]) -> httpx.Response:
                response = self.client.post(
                    url="/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                return response

            batch_responses = [
                _embed_batch(batch) for batch in self._get_batches(texts)
            ]
            return [
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception:
            logger.exception("An error occurred with OpenRouter")
            raise

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts asynchronously.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        try:

            @self._retry
            async def _aembed_batch(batch: list[str]) -> httpx.Response:
                response = await self.async_client.post(
                    url="/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                return response

            batch_responses = await asyncio.gather(
                *[_aembed_batch(batch) for batch in self._get_batches(texts)]
            )
            return [
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception:
            logger.exception("An error occurred with OpenRouter")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.

        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text asynchronously.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.

        """
        return (await self.aembed_documents([text]))[0]
