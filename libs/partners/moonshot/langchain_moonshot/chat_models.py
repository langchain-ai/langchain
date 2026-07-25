"""Moonshot chat models."""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlparse

import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LangSmithParams, ModelProfile, ModelProfileRegistry
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_moonshot._version import __version__
from langchain_moonshot.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://api.moonshot.cn/v1"

_DictOrPydanticClass: type = dict[str, Any] | type[BaseModel]
_DictOrPydantic: type = dict[str, Any] | BaseModel

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatMoonshot(BaseChatOpenAI):
    """Moonshot chat model integration.

    Moonshot AI provides powerful large language models through its API.
    This integration supports chat, streaming, tool calling, and structured output.

    Setup:
        Install ``langchain-moonshot`` and set environment variable ``MOONSHOT_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-moonshot
            export MOONSHOT_API_KEY="your-api-key"

    Key init args - completion params:
        model:
            Name of Moonshot model to use, e.g. ``moonshot-v1-8k``.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

    Key init args - client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Moonshot API key. If not passed in will be read from env var ``MOONSHOT_API_KEY``.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_moonshot import ChatMoonshot

            model = ChatMoonshot(
                model="moonshot-v1-8k",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
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
                \"\"\"Get the current weather in a given location\"\"\"
                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            model_with_tools = model.bind_tools([GetWeather])
            ai_msg = model_with_tools.invoke("What is the weather in San Francisco?")
            ai_msg.tool_calls

    Structured output:
        .. code-block:: python

            from typing import Optional
            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                \"\"\"Joke to tell user.\"\"\"
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

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
    """

    model_name: str = Field(alias="model")
    """The name of the model"""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("MOONSHOT_API_KEY", default=None),
    )
    """Moonshot API key"""

    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("MOONSHOT_API_BASE", default=DEFAULT_API_BASE),
    )
    """Moonshot API base URL.

    Automatically read from env variable ``MOONSHOT_API_BASE`` if not provided.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-moonshot"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "MOONSHOT_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "moonshot"
        return ls_params

    @model_validator(mode="after")
    def _set_moonshot_version(self) -> Self:
        """Set package version in metadata."""
        self._add_version("langchain-moonshot", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, MOONSHOT_API_KEY must be set."
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

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    def _get_request_payload(
        self,
        input_: BaseMessage | list[BaseMessage],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message in payload["messages"]:
            if message["role"] == "tool" and isinstance(message["content"], list):
                message["content"] = __import__("json").dumps(message["content"])
            elif message["role"] == "assistant" and isinstance(message["content"], list):
                text_parts = [
                    block.get("text", "")
                    for block in message["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message["content"] = "".join(text_parts) if text_parts else ""
        return payload

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
            generation.message.response_metadata["model_provider"] = "moonshot"

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk and isinstance(generation_chunk.message, BaseModel):
            if generation_chunk.message.response_metadata is None:
                generation_chunk.message.response_metadata = {}
            generation_chunk.message.response_metadata["model_provider"] = "moonshot"
        return generation_chunk

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "streaming": self.streaming,
        }
