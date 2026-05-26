"""Atlas Cloud chat models."""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.language_models import LangSmithParams
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_API_BASE = "https://api.atlascloud.ai/v1"


class ChatAtlas(BaseChatOpenAI):
    """Atlas Cloud chat model integration via the OpenAI-compatible API.

    Setup:
        Install `langchain-atlas` and set environment variable `ATLAS_API_KEY`.

        ```bash
        pip install -U langchain-atlas
        export ATLAS_API_KEY="your-api-key"
        ```

    Instantiate:
        ```python
        from langchain_atlas import ChatAtlas

        model = ChatAtlas(model="deepseek-ai/DeepSeek-V3-0324", temperature=0)
        ```

    Invoke:
        ```python
        model.invoke("Say hello in Chinese.")
        ```

    Stream:
        ```python
        for chunk in model.stream("Count from one to three."):
            print(chunk.text, end="")
        ```
    """

    model_name: str = Field(default="deepseek-ai/DeepSeek-V3-0324", alias="model")
    """The Atlas model name."""

    atlas_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("ATLAS_API_KEY", default=None),
    )
    """Atlas Cloud API key."""

    atlas_api_base: str = Field(
        alias="base_url",
        default_factory=from_env("ATLAS_API_BASE", default=DEFAULT_API_BASE),
    )
    """Atlas Cloud API base URL."""

    openai_api_key: SecretStr | None = None
    openai_api_base: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "atlas-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"atlas_api_key": "ATLAS_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object."""
        return ["langchain_atlas", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get LangSmith tracing params."""
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "atlas"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment configuration and initialize clients."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        client_params: dict[str, Any] = {
            "api_key": (
                self.atlas_api_key.get_secret_value() if self.atlas_api_key else None
            ),
            "base_url": self.atlas_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            msg = (
                "Atlas API key is not set. Please set it in the `atlas_api_key` field "
                "or in the `ATLAS_API_KEY` environment variable."
            )
            raise ValueError(msg)

        if not (self.client or None):
            sync_specific: dict[str, Any] = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict[str, Any] = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions

        if self.stream_usage is not False:
            self.stream_usage = True

        return self
