"""Sarvam chat model integration."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import requests
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from pydantic import Field, SecretStr, model_validator

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a Sarvam API message dict."""
    message_dict: dict[str, Any] = {}

    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
    else:
        raise ValueError(f"Got unknown message type: {message}")

    message_dict["content"] = message.content
    return message_dict


def _convert_dict_to_message(response: dict) -> BaseMessage:
    """Convert a Sarvam API response to a LangChain message."""
    role = response.get("role", "assistant")
    content = response.get("content", "")

    if role == "assistant":
        return AIMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return ChatMessage(content=content, role=role)


class ChatSarvam(BaseChatModel):
    """Sarvam AI chat model integration.

    Setup:
        Install ``langchain-sarvam`` and set environment variable ``SARVAM_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-sarvam
            export SARVAM_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Sarvam model to use.
        temperature: float
            Sampling temperature.
        max_tokens: int | None
            Max number of tokens to generate.

    Key init args — client params:
        api_key: SecretStr | None
            Sarvam API key. If not provided, will read from SARVAM_API_KEY env var.
        base_url: str
            Base URL for Sarvam API.

    Instantiate:
        .. code-block:: python

            from langchain_sarvam import ChatSarvam

            llm = ChatSarvam(
                model="sarvam-m",
                temperature=0.7,
                max_tokens=1024,
                # api_key="your-api-key",
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "What is the capital of France?"),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='The capital of France is Paris.')

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

        .. code-block:: python

            The capital of France is Paris.

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

        .. code-block:: python

            AIMessage(content='The capital of France is Paris.')

    """

    model: str = Field(default="sarvam-m")
    """Model name to use."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    """Sampling temperature."""

    max_tokens: int | None = Field(default=None)
    """Maximum number of tokens to generate."""

    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    """Nucleus sampling parameter."""

    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    """Penalize new tokens based on their frequency in the text so far."""

    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    """Penalize new tokens based on whether they appear in the text so far."""

    n: int = Field(default=1, ge=1)
    """Number of chat completions to generate for each prompt."""

    streaming: bool = False
    """Whether to stream the results or not."""

    base_url: str = Field(default="https://api.sarvam.ai/v1")
    """Base URL for Sarvam API."""

    api_key: SecretStr | None = Field(default=None)
    """Sarvam API key."""

    timeout: float | None = Field(default=None)
    """Timeout for API requests in seconds."""

    max_retries: int = Field(default=2, ge=0)
    """Maximum number of retries for API requests."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional model parameters."""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key exists in environment."""
        values["api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "api_key", "SARVAM_API_KEY")
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "sarvam-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            **self.model_kwargs,
        }

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling Sarvam API."""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "n": self.n,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        return params

    def _create_chat_result(self, response: dict[str, Any]) -> ChatResult:
        """Create a ChatResult from a Sarvam API response."""
        generations = []
        for choice in response.get("choices", []):
            message_dict = choice.get("message", {})
            message = _convert_dict_to_message(message_dict)
            generation = ChatGeneration(
                message=message,
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    index=choice.get("index"),
                ),
            )
            generations.append(generation)

        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": response.get("model", self.model),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._default_params
        params.update(kwargs)
        if stop is not None:
            params["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",  # type: ignore[union-attr]
            "Content-Type": "application/json",
        }

        payload = {
            "messages": message_dicts,
            **params,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        # Add better error handling
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = response.json()
                logger.error(f"Sarvam API error: {error_detail}")
            except Exception:
                error_detail = response.text
                logger.error(f"Sarvam API error: {error_detail}")
            raise ValueError(f"Sarvam API request failed: {e}. Details: {error_detail}")

        return self._create_chat_result(response.json())

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._default_params
        params.update(kwargs)
        params["stream"] = True
        if stop is not None:
            params["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",  # type: ignore[union-attr]
            "Content-Type": "application/json",
        }

        payload = {
            "messages": message_dicts,
            **params,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        import json

                        data = json.loads(data_str)
                        for choice in data.get("choices", []):
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=content)
                                )
                                if run_manager:
                                    run_manager.on_llm_new_token(content, chunk=chunk)
                                yield chunk
                    except Exception as e:
                        logger.warning(f"Error parsing stream: {e}")
                        continue

    @property
    def _invocation_params(self) -> dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params
