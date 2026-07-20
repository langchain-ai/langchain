"""OpenRouter Rerank Model."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.utils import from_env, secret_from_env
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self


class OpenRouterRerank(BaseDocumentCompressor):
    """OpenRouter rerank model integration.

    OpenRouter is a unified API that provides access to various rerank models from
    multiple providers (NVIDIA, Cohere, Jina, etc.).

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
        | `model` | `str` | Model name |
        | `top_n` | `int` | Number of documents to return. Default is `3`. |
        | `openrouter_provider` | `dict[str, Any] | None` | Routing preferences. |
        | `route` | `str | None` | Routing strategy (e.g. 'fallback'). |

    ??? info "Key init args — client params"

        | Param | Type | Description |
        | ----- | ---- | ----------- |
        | `api_key` | `str | None` | OpenRouter API key. |
        | `base_url` | `str | None` | Base URL for API requests. |
        | `app_url` | `str | None` | App URL for attribution. |
        | `app_title` | `str | None` | App title for attribution. |
        | `app_categories` | `list[str] | None` | Marketplace attribution categories. |
        | `default_headers` | `Mapping[str, str] | None` | Extra request headers. |
        | `max_retries` | `int` | Max retries (default `2`). Set to `0` to disable. |
        | `request_timeout` | `int | None` | Timeout in milliseconds. |

    ??? info "Instantiate"

        ```python
        from langchain_openrouter import OpenRouterReranker

        reranker = OpenRouterReranker(
            model="cohere/rerank-v3.5",
            top_n=5,
            # api_key="...",
        )
        ```

    See https://openrouter.ai/docs/rerank for platform documentation.
    """

    client: Any = Field(default=None, exclude=True)
    """SDK Client (`openrouter.OpenRouter`)"""

    api_base_ulr: str | None = Field(
        default_factory=from_env("OPENROUTER_API_BASE", default=None), alias="base_url"
    )
    """Base URL for API requests.

    Defaults to OpenRouter API base URL.

    See https://openrouter.ai/docs for details.
    """

    openrouter_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    """OpenRouter API key.

    See https://openrouter.ai/docs/api-key for details.
    """

    app_url: str | None = Field(
        default_factory=from_env(
            "OPENROUTER_APP_URL", default="https://docs.langchain.com"
        ),
        alias="app_url",
    )
    """Application URL for OpenRouter attribution.

    Maps to `HTTP-Referer` header.

    Defaults to LangChain docs URL. Set this to your app's URL to get
    attribution for API usage in the OpenRouter dashboard.

    See https://openrouter.ai/docs/app-attribution for details.
    """

    app_title: str | None = Field(
        default_factory=from_env("OPENROUTER_APP_TITLE", default="LangChain"),
        alias="app_title",
    )
    """Application title for OpenRouter attribution.

    Maps to `X-Title` header.

    Defaults to `'LangChain'`. Set this to your app's name to get attribution
    for API usage in the OpenRouter dashboard.

    See https://openrouter.ai/docs/app-attribution for details.
    """

    app_categories: list[str] | None = Field(default=None)
    """Marketplace categories for OpenRouter attribution.

    Maps to `X-OpenRouter-Categories` header. Pass a list of lowercase,
    hyphen-separated category strings (max 30 characters each),
    e.g. `['cli-agent', 'programming-app']`.

    Only recognized categories are accepted (unrecognized values are silently
    dropped by OpenRouter).

    See https://openrouter.ai/docs/app-attribution for recognized categories.
    """

    request_timeout: int | None = Field(default=None, alias="timeout")
    """Timeout for requests in milliseconds. Maps to SDK `timeout`"""

    max_retries: int = 2
    """Maximum number of retries.

    Each unit adds ~150 seconds to the backoff window via the SDK's
    `max_elapsed_time` (e.g. `max_retries=2` allows up to ~300 s).

    Set to `0` to disable retries.
    """

    top_n: int = Field(default=3)
    """Number of most relevant documents to return"""

    model: str = Field(..., alias="model_name")
    """The rerank model to use.
    e.g. cohere/rerank-v3.5
    """
    openrouter_provider: dict[str, Any] | None = None
    """Provider preferences to pass to OpenRouter.

    Example: `{"order": ["Anthropic", "OpenAI"]}`
    """

    default_headers: Mapping[str, str] | None = Field(default=None, exclude=True)
    """Additional HTTP headers to include on every request to OpenRouter.

    Headers set via this field are merged with the OpenRouter app-attribution
    headers (`HTTP-Referer`, `X-Title`, `X-OpenRouter-Categories`); on a
    collision the value from `default_headers` takes precedence.
    """

    route: str | None = None
    """Route preference for OpenRouter, e.g. `'fallback'`."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Any extra model parameters for the OpenRouter API."""

    def _build_client(self) -> Any:
        """Build and return an `openrouter.OpenRouter` SDK client.

        Returns:
            An `openrouter.OpenRouter` SDK client instance.
        """
        import openrouter  # noqa: PLC0415
        from openrouter.utils import (  # noqa: PLC0415
            BackoffStrategy,
            RetryConfig,
        )

        client_kwargs: dict[str, Any] = {
            "api_key": (
                self.openrouter_api_key.get_secret_value()  # type: ignore[union-attr]
            )
        }

        if self.api_base_ulr:
            client_kwargs["server"] = self.api_base_ulr
        extra_headers: dict[str, Any] = {}

        if self.app_title is not None:
            extra_headers["X-Title"] = self.app_title
        if self.app_url is not None:
            extra_headers["HTTP-Referer"] = self.app_url
        if self.app_categories is not None:
            extra_headers["X-OpenRouter-Categories"] = ",".join(self.app_categories)

        if self.default_headers:
            user_header_names = {name.casefold() for name in self.default_headers}
            extra_headers = {
                name: value
                for name, value in extra_headers.items()
                if name.casefold() not in user_header_names
            }
            extra_headers.update(self.default_headers)

        if extra_headers:
            client_kwargs["client"] = httpx.Client(
                headers=extra_headers, follow_redirects=True
            )
            client_kwargs["async_client"] = httpx.AsyncClient(
                headers=extra_headers, follow_redirects=True
            )
        if self.request_timeout:
            client_kwargs["timeout_ms"] = self.request_timeout
        if self.max_retries > 0:
            client_kwargs["retry_config"] = RetryConfig(
                strategy="backoff",
                backoff=BackoffStrategy(
                    initial_interval=500,
                    max_interval=60000,
                    exponent=1.5,
                    max_elapsed_time=self.max_retries * 150_000,
                ),
                retry_connection_errors=True,
            )
        return openrouter.OpenRouter(**client_kwargs)

    @model_validator(mode="after")
    def validate_envoirenment(self) -> Self:
        """Validate configuration and build the SDK client."""
        if not (self.openrouter_api_key and self.openrouter_api_key.get_secret_value()):
            msg = "OPENROUTER_API_KEY must be set."
            raise ValueError(msg)

        if not self.client:
            try:
                self.client = self._build_client()
            except ImportError as e:
                msg = (
                    "Could not import the `openrouter` Python SDK. "
                    "Please install it with: pip install openrouter"
                )
                raise ImportError(msg) from e
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of rerank model."""
        return "openrouter-rerank"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "top_n": self.top_n,
            "openrouter_provider": self.openrouter_provider,
            "route": self.route,
            "model_kwargs": self.model_kwargs,
        }

    @staticmethod
    def _serialize_documents(
        documents: Sequence[Document],
    ) -> list[dict[str, Any] | str]:
        """Serialize documents for OpenRouter rerank model."""
        serialized: list[dict[str, Any] | str] = []
        for d in documents:
            image_url = d.metadata.get("image_url")

            if image_url:
                serialized.append({"text": d.page_content, "image": image_url})
            else:
                serialized.append(d.page_content)

        return serialized

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,  # noqa: ARG002
    ) -> list[Document]:
        """Compress documents using OpenRouter rerank model.

        Args:
            documents: Documents to rerank.
            query: Query to rerank documents against.
            callbacks: Callbacks to use for the reranking.

        Return:
            List of reranked documents.

        Raises:
            ValueError: If no documents are provided.
        """
        if not documents:
            msg = "No documents provided for reranking."
            raise ValueError(msg)

        serialized_docs = self._serialize_documents(documents)

        response = self.client.rerank.rerank(
            model=self.model,
            documents=serialized_docs,
            query=query,
            top_n=self.top_n,
            **self.model_kwargs,
        )

        ranked_documents = []
        for r in response.results:
            index = r.index
            original_document = documents[index]

            original_document.metadata["relevance_score"] = r.relevance_score

            ranked_documents.append(original_document)

        return ranked_documents
