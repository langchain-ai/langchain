"""Wrapper around xAI's Chat Completions API."""

import os
from typing import Any, Dict, List, Optional

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.utils import secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class ChatXAI(BaseChatOpenAI):
    """Chat model for xAI's chat completions API, supporting Grok models.

    This class provides a way to interact with xAI's chat models, including
    versioned Grok models (e.g., "grok-2", "grok-3") via the `grok_version`
    parameter.
    """

    model_name: str = Field(default="grok-beta", alias="model")
    """Model name to use."""
    xai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("XAI_API_KEY", default=None),
    )
    """xAI API key. Auto-read from `XAI_API_KEY` env var if not provided."""
    xai_api_base: str = Field(default="https://api.x.ai/v1/")
    """Base URL path for API requests."""

    openai_api_key: Optional[SecretStr] = None
    openai_api_base: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-beta",
        grok_version: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize ChatXAI with xAI API key and optional Grok version.

        Args:
            api_key: xAI API key (defaults to `XAI_API_KEY` env var).
            model: Model name (e.g., "grok-beta", overridden by grok_version).
            grok_version: Optional Grok version (e.g., "2" for "grok-2").
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            max_retries: Max retries for API requests.
            **kwargs: Additional arguments passed to BaseChatOpenAI.
        """
        api_key_str = api_key or os.environ.get("XAI_API_KEY")
        if not api_key_str:
            raise ValueError("XAI_API_KEY must be provided or set in environment.")
        if grok_version:
            model = f"grok-{grok_version}"
        super().__init__(
            api_key=SecretStr(api_key_str) if api_key_str else None,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs,
        )

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"xai_api_key": "XAI_API_KEY"}
        """
        return {"xai_api_key": "XAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_xai", "chat_models"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: Dict[str, Any] = {}

        if self.xai_api_base:
            attributes["xai_api_base"] = self.xai_api_base

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "xai-chat"

    def _get_ls_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "xai"
        return params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params: Dict[str, Any] = {
            "api_key": (
                self.xai_api_key.get_secret_value() if self.xai_api_key else None
            ),
            "base_url": self.xai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            raise ValueError(
                "XAI_API_KEY must be set in the `xai_api_key` field or "
                "in the `XAI_API_KEY` environment variable."
            )

        if not (self.client or None):
            sync_specific: Dict[str, Any] = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params,
                **sync_specific,
            ).chat.completions
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
        if not (self.async_client or None):
            async_specific: Dict[str, Any] = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            ).chat.completions
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
        return self
