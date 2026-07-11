"""Requesty chat models."""

from __future__ import annotations

from typing import Any, cast

import openai
from langchain_core.language_models import (
    LangSmithParams,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.outputs import ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_requesty._version import __version__
from langchain_requesty.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://router.requesty.ai/v1"

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatRequesty(BaseChatOpenAI):
    """Requesty chat model integration.

    Requesty is an OpenAI-compatible LLM gateway that provides access to models
    from many providers (OpenAI, Anthropic, Google, DeepSeek, etc.) through a
    single endpoint, using ``provider/model`` naming (e.g.
    ``openai/gpt-4o-mini``, ``anthropic/claude-sonnet-4-5``).

    Setup:
        Install `langchain-requesty` and set environment variable
        `REQUESTY_API_KEY`.

        ```bash
        pip install -U langchain-requesty
        export REQUESTY_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of Requesty model to use, e.g. `'openai/gpt-4o-mini'`.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Requesty API key. If not passed in will be read from env var
            `REQUESTY_API_KEY`.
        app_url:
            Optional app URL for attribution. Maps to the `HTTP-Referer` header.
        app_title:
            Optional app title for attribution. Maps to the `X-Title` header.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        ```python
        from langchain_requesty import ChatRequesty

        model = ChatRequesty(
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city, e.g. San Francisco")


        model_with_tools = model.bind_tools([GetWeather])
        ai_msg = model_with_tools.invoke("What is the weather in San Francisco?")
        ai_msg.tool_calls
        ```

        See `ChatRequesty.bind_tools()` method for more.

    Structured output:
        ```python
        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        See `ChatRequesty.with_structured_output()` for more.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```

    See https://docs.requesty.ai for platform documentation.
    """

    model_name: str = Field(alias="model")
    """The name of the model, e.g. `'openai/gpt-4o-mini'`."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("REQUESTY_API_KEY", default=None),
    )
    """Requesty API key."""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("REQUESTY_API_BASE", default=DEFAULT_API_BASE),
    )
    """Requesty API base URL.

    Automatically read from env variable `REQUESTY_API_BASE` if not provided.
    Defaults to the Requesty OpenAI-compatible router endpoint.
    """
    app_url: str | None = Field(
        default_factory=from_env("REQUESTY_APP_URL", default=None),
    )
    """Application URL for Requesty attribution.

    Maps to the `HTTP-Referer` header. Set this to your app's URL to get
    attribution for API usage in the Requesty dashboard.
    """
    app_title: str | None = Field(
        default_factory=from_env("REQUESTY_APP_TITLE", default=None),
    )
    """Application title for Requesty attribution.

    Maps to the `X-Title` header. Set this to your app's name to get
    attribution for API usage in the Requesty dashboard.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-requesty"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "REQUESTY_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "requesty"
        return ls_params

    @model_validator(mode="after")
    def _set_requesty_version(self) -> Self:
        """Set package version in metadata.

        Named uniquely to avoid shadowing `BaseChatOpenAI._set_openai_chat_version`;
        Pydantic replaces same-named validators rather than chaining them.
        """
        self._add_version("langchain-requesty", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, REQUESTY_API_KEY must be set."
            raise ValueError(msg)

        # Merge optional attribution headers (HTTP-Referer / X-Title) with any
        # user-supplied default headers. User-supplied values take precedence.
        attribution_headers: dict[str, str] = {}
        if self.app_url:
            attribution_headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            attribution_headers["X-Title"] = self.app_title
        default_headers = self.default_headers
        if attribution_headers:
            merged = dict(attribution_headers)
            if default_headers:
                merged.update(default_headers)
            default_headers = merged

        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": default_headers,
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

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "requesty"

        return rtn
