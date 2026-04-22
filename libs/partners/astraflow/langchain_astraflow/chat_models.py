"""Astraflow chat models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import openai
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_astraflow.data._profiles import _PROFILES

if TYPE_CHECKING:
    from langchain_core.language_models import (
        ModelProfile,
        ModelProfileRegistry,
    )
    from langchain_core.language_models.chat_models import LangSmithParams

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)

# Global endpoint (default)
_ASTRAFLOW_API_BASE = "https://api-us-ca.umodelverse.ai/v1"


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatAstraflow(BaseChatOpenAI):  # type: ignore[override]
    r"""ChatAstraflow chat model.

    Astraflow (by UCloud / 优刻得) is an OpenAI-compatible AI model aggregation
    platform supporting 200+ models including GPT, Claude, Gemini, DeepSeek,
    Qwen, and many more.

    Two regional endpoints are available:

    - **Global:** ``https://api-us-ca.umodelverse.ai/v1`` — env var ``ASTRAFLOW_API_KEY``
    - **China:**  ``https://api.modelverse.cn/v1``         — env var ``ASTRAFLOW_CN_API_KEY``

    Setup:
        Install ``langchain-astraflow`` and set the ``ASTRAFLOW_API_KEY`` environment
        variable (global endpoint) or ``ASTRAFLOW_CN_API_KEY`` (China endpoint).

        .. code-block:: bash

            pip install -U langchain-astraflow
            export ASTRAFLOW_API_KEY="your-api-key"

    Key init args — completion params:
        model:
            Name of the model to use (e.g. ``'gpt-4o'``, ``'deepseek-v3'``,
            ``'claude-3-5-sonnet'``).
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum number of tokens to generate.

    Key init args — client params:
        api_key:
            Astraflow API key. If not provided, read from ``ASTRAFLOW_API_KEY``.
        base_url:
            API base URL. Defaults to the global endpoint
            ``https://api-us-ca.umodelverse.ai/v1``.
            Override via env var ``ASTRAFLOW_API_BASE``.
        timeout:
            Timeout for requests.
        max_retries:
            Maximum number of retries.

    Instantiate:
        .. code-block:: python

            from langchain_astraflow import ChatAstraflow

            model = ChatAstraflow(
                model="gpt-4o",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "What is the capital of France?"),
            ]
            model.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in model.stream(messages):
                print(chunk.text, end="")

    China endpoint:
        .. code-block:: python

            model_cn = ChatAstraflow(
                model="deepseek-v3",
                base_url="https://api.modelverse.cn/v1",
                api_key="your-cn-api-key",
            )

    See https://umodelverse.ai for platform documentation.
    """

    model_name: str = Field(default="gpt-4o", alias="model")
    """Model name to use."""

    astraflow_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("ASTRAFLOW_API_KEY", default=None),
    )
    """Astraflow API key (global endpoint).

    Automatically read from the ``ASTRAFLOW_API_KEY`` environment variable if
    not provided.  For the China endpoint, set ``ASTRAFLOW_CN_API_KEY`` and
    pass the value here (or set ``base_url`` to the China endpoint).
    """

    astraflow_api_base: str = Field(
        alias="base_url",
        default_factory=from_env(
            "ASTRAFLOW_API_BASE",
            default=_ASTRAFLOW_API_BASE,
        ),
    )
    """Astraflow API base URL.

    Defaults to the global endpoint ``https://api-us-ca.umodelverse.ai/v1``.
    Override via the ``ASTRAFLOW_API_BASE`` environment variable or by passing
    ``base_url`` directly.

    For the China endpoint use ``https://api.modelverse.cn/v1``.
    """

    openai_api_key: SecretStr | None = None
    openai_api_base: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"astraflow_api_key": "ASTRAFLOW_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object."""
        return ["langchain_astraflow", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "astraflow-chat"

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "astraflow"
        return params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exist in environment."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        client_params: dict = {
            "api_key": (
                self.astraflow_api_key.get_secret_value()
                if self.astraflow_api_key
                else None
            ),
            "base_url": self.astraflow_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            msg = (
                "Astraflow API key is not set. Please set it in the "
                "`astraflow_api_key` field or in the `ASTRAFLOW_API_KEY` "
                "environment variable."
            )
            raise ValueError(msg)

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )

        if self.stream_usage is not False:
            self.stream_usage = True

        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None
