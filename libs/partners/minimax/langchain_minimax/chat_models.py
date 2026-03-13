"""MiniMax chat models."""

from __future__ import annotations

from typing import Any, cast

import openai
from langchain_core.language_models import (
    LangSmithParams,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_minimax.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://api.minimax.io/v1"

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatMiniMax(BaseChatOpenAI):
    """MiniMax chat model integration to access models hosted on MiniMax's API.

    Setup:
        Install ``langchain-minimax`` and set environment variable
        ``MINIMAX_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-minimax
            export MINIMAX_API_KEY="your-api-key"

    Key init args --- completion params:
        model:
            Name of MiniMax model to use, e.g. ``'MiniMax-M2.5'``.
        temperature:
            Sampling temperature. Must be in (0.0, 1.0].
        max_tokens:
            Max number of tokens to generate.

    Key init args --- client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            MiniMax API key. If not passed in will be read from env var
            ``MINIMAX_API_KEY``.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        .. code-block:: python

            from langchain_minimax import ChatMiniMax

            model = ChatMiniMax(
                model="MiniMax-M2.5",
                temperature=0.5,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful translator. Translate the user sentence to French.",
                ),
                ("human", "I love programming."),
            ]
            model.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in model.stream(messages):
                print(chunk.text, end="")

    Async:
        .. code-block:: python

            await model.ainvoke(messages)

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ...,
                    description="The city and state, e.g. San Francisco, CA",
                )


            model_with_tools = model.bind_tools([GetWeather])
            ai_msg = model_with_tools.invoke("What is the weather like in San Francisco?")
            ai_msg.tool_calls

    Structured output:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: int | None = Field(description="How funny the joke is, from 1 to 10")


            structured_model = model.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats")

    Token usage:
        .. code-block:: python

            ai_msg = model.invoke(messages)
            ai_msg.usage_metadata

    Response metadata:
        .. code-block:: python

            ai_msg = model.invoke(messages)
            ai_msg.response_metadata
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The name of the model."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("MINIMAX_API_KEY", default=None),
    )
    """MiniMax API key."""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("MINIMAX_API_BASE", default=DEFAULT_API_BASE),
    )
    """MiniMax API base URL.

    Automatically read from env variable ``MINIMAX_API_BASE`` if not provided.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-minimax"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "MINIMAX_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "minimax"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, MINIMAX_API_KEY must be set."
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

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model_name)
        return self
