"""HPC-AI chat models (OpenAI-compatible API)."""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.language_models import LangSmithParams
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_API_BASE = "https://api.hpc-ai.com/inference/v1"


class ChatHPCAI(BaseChatOpenAI):
    """HPC-AI chat model via an OpenAI-compatible HTTP API.

    Setup:
        Install `langchain-hpc-ai` and set environment variable `HPC_AI_API_KEY`.

        ```bash
        pip install -U langchain-hpc-ai
        export HPC_AI_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Model identifier, e.g. `'minimax/minimax-m2.5'` or `'moonshotai/kimi-k2.5'`.
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
            HPC-AI API key. If not passed in will be read from env var `HPC_AI_API_KEY`.
        base_url:
            API base URL. Defaults to the HPC-AI inference endpoint; override with env
            `HPC_AI_BASE_URL` if needed.

    Instantiate:
        ```python
        from langchain_hpc_ai import ChatHPCAI

        model = ChatHPCAI(
            model="minimax/minimax-m2.5",
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
            ("system", "You are a helpful assistant."),
            ("human", "Hello!"),
        ]
        model.invoke(messages)
        ```
    """

    model_name: str = Field(alias="model")
    """The name of the model."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("HPC_AI_API_KEY", default=None),
    )
    """HPC-AI API key."""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("HPC_AI_BASE_URL", default=DEFAULT_API_BASE),
    )
    """HPC-AI API base URL.

    Automatically read from env variable `HPC_AI_BASE_URL` if not provided.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-hpc-ai"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "HPC_AI_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "hpc_ai"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, HPC_AI_API_KEY must be set."
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
            generation.message.response_metadata["model_provider"] = "hpc_ai"

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
        if generation_chunk:
            generation_chunk.message.response_metadata = {
                **generation_chunk.message.response_metadata,
                "model_provider": "hpc_ai",
            }
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                reasoning_content = top.get("delta", {}).get("reasoning_content")
                if reasoning_content is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )

        return generation_chunk
