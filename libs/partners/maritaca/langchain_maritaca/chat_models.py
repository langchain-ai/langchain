"""Maritaca AI Chat wrapper for LangChain.

Maritaca AI provides Brazilian Portuguese-optimized language models,
including the Sabiá family of models.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_maritaca.version import __version__

# HTTP status code for rate limiting
HTTP_TOO_MANY_REQUESTS = 429


class ChatMaritaca(BaseChatModel):
    r"""Maritaca AI Chat large language models API.

    Maritaca AI provides Brazilian Portuguese-optimized language models,
    offering excellent performance for Portuguese text generation, analysis,
    and understanding tasks.

    To use, you should have the environment variable `MARITACA_API_KEY`
    set with your API key, or pass it as a named parameter to the constructor.

    Setup:
        Install `langchain-maritaca` and set environment variable
        `MARITACA_API_KEY`.

        ```bash
        pip install -U langchain-maritaca
        export MARITACA_API_KEY="your-api-key"
        ```

    Key init args - completion params:
        model:
            Name of Maritaca model to use. Available models:
            - `sabia-3` (default): Most capable model
            - `sabiazinho-3`: Faster and more economical
        temperature:
            Sampling temperature. Ranges from 0.0 to 2.0.
        max_tokens:
            Max number of tokens to generate.

    Key init args - client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Maritaca API key. If not passed in will be read from
            env var `MARITACA_API_KEY`.

    Instantiate:
        ```python
        from langchain_maritaca import ChatMaritaca

        model = ChatMaritaca(
            model="sabia-3",
            temperature=0.7,
            max_retries=2,
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "Você é um assistente prestativo."),
            ("human", "Qual é a capital do Brasil?"),
        ]
        model.invoke(messages)
        ```
        ```python
        AIMessage(
            content="A capital do Brasil é Brasília.",
            response_metadata={"model": "sabia-3", "finish_reason": "stop"},
        )
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

    Async:
        ```python
        await model.ainvoke(messages)
        ```
    """

    client: Any = Field(default=None, exclude=True)
    """Sync HTTP client."""

    async_client: Any = Field(default=None, exclude=True)
    """Async HTTP client."""

    model_name: str = Field(default="sabia-3", alias="model")
    """Model name to use.

    Available models:
    - sabia-3: Most capable model (R$ 5.00/R$ 10.00 per 1M tokens)
    - sabiazinho-3: Fast and economical (R$ 1.00/R$ 3.00 per 1M tokens)
    """

    temperature: float = 0.7
    """Sampling temperature (0.0 to 2.0)."""

    max_tokens: int | None = Field(default=None)
    """Maximum number of tokens to generate."""

    top_p: float = 0.9
    """Top-p sampling parameter."""

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    frequency_penalty: float = 0.0
    """Frequency penalty (-2.0 to 2.0)."""

    presence_penalty: float = 0.0
    """Presence penalty (-2.0 to 2.0)."""

    maritaca_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("MARITACA_API_KEY", default=None),
    )
    """Maritaca API key. Automatically inferred from env var `MARITACA_API_KEY`."""

    maritaca_api_base: str = Field(
        alias="base_url",
        default_factory=from_env(
            "MARITACA_API_BASE", default="https://chat.maritaca.ai/api"
        ),
    )
    """Base URL for Maritaca API."""

    request_timeout: float | None = Field(default=60.0, alias="timeout")
    """Timeout for requests in seconds."""

    max_retries: int = 2
    """Maximum number of retries."""

    streaming: bool = False
    """Whether to stream results."""

    n: int = 1
    """Number of completions to generate."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that API key exists and initialize HTTP clients."""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        # Ensure temperature is not exactly 0 (causes issues with some APIs)
        if self.temperature == 0:
            self.temperature = 1e-8

        # Initialize HTTP clients
        api_key = (
            self.maritaca_api_key.get_secret_value() if self.maritaca_api_key else ""
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"langchain-maritaca/{__version__}",
        }

        if not self.client:
            self.client = httpx.Client(
                base_url=self.maritaca_api_base,
                headers=headers,
                timeout=httpx.Timeout(self.request_timeout),
            )

        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=self.maritaca_api_base,
                headers=headers,
                timeout=httpx.Timeout(self.request_timeout),
            )

        return self

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret environment variables."""
        return {"maritaca_api_key": "MARITACA_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "maritaca-chat"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="maritaca",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling Maritaca API."""
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop is not None:
            params["stop"] = self.stop
        return params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion."""
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        response = self._make_request(message_dicts, params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a chat completion."""
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        response = await self._amake_request(message_dicts, params)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a chat completion."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        with self.client.stream(
            "POST",
            "/chat/completions",
            json={"messages": message_dicts, **params},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if not chunk.get("choices"):
                            continue
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        message_chunk = AIMessageChunk(content=content)
                        generation_info = {}

                        if finish_reason := choice.get("finish_reason"):
                            generation_info["finish_reason"] = finish_reason
                            generation_info["model"] = self.model_name

                        if generation_info:
                            message_chunk = message_chunk.model_copy(
                                update={"response_metadata": generation_info}
                            )

                        generation_chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info or None,
                        )

                        if run_manager:
                            run_manager.on_llm_new_token(
                                generation_chunk.text, chunk=generation_chunk
                            )

                        yield generation_chunk

                    except json.JSONDecodeError:
                        continue

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream a chat completion."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        async with self.async_client.stream(
            "POST",
            "/chat/completions",
            json={"messages": message_dicts, **params},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if not chunk.get("choices"):
                            continue
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        message_chunk = AIMessageChunk(content=content)
                        generation_info = {}

                        if finish_reason := choice.get("finish_reason"):
                            generation_info["finish_reason"] = finish_reason
                            generation_info["model"] = self.model_name

                        if generation_info:
                            message_chunk = message_chunk.model_copy(
                                update={"response_metadata": generation_info}
                            )

                        generation_chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info or None,
                        )

                        if run_manager:
                            await run_manager.on_llm_new_token(
                                token=generation_chunk.text, chunk=generation_chunk
                            )

                        yield generation_chunk

                    except json.JSONDecodeError:
                        continue

    def _make_request(
        self, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a sync request to Maritaca API."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    "/chat/completions",
                    json={"messages": messages, **params},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                is_rate_limited = e.response.status_code == HTTP_TOO_MANY_REQUESTS
                if is_rate_limited and attempt < self.max_retries:
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    time.sleep(retry_after)
                    continue
                raise
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    continue
                raise
        msg = f"Failed after {self.max_retries + 1} attempts"
        raise RuntimeError(msg)

    async def _amake_request(
        self, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Make an async request to Maritaca API."""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.async_client.post(
                    "/chat/completions",
                    json={"messages": messages, **params},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                is_rate_limited = e.response.status_code == HTTP_TOO_MANY_REQUESTS
                if is_rate_limited and attempt < self.max_retries:
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    await asyncio.sleep(retry_after)
                    continue
                raise
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    continue
                raise
        msg = f"Failed after {self.max_retries + 1} attempts"
        raise RuntimeError(msg)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Convert LangChain messages to Maritaca format."""
        params = self._default_params.copy()
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: dict[str, Any]) -> ChatResult:
        """Create a ChatResult from Maritaca API response."""
        generations = []
        token_usage = response.get("usage", {})

        for choice in response.get("choices", []):
            message = _convert_dict_to_message(choice.get("message", {}))

            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)

            generation_info = {"finish_reason": choice.get("finish_reason")}
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model": response.get("model", self.model_name),
        }

        return ChatResult(generations=generations, llm_output=llm_output)


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to Maritaca format.

    Args:
        message: The LangChain message.

    Returns:
        Dictionary in Maritaca API format.
    """
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    msg = f"Got unknown message type: {type(message)}"
    raise TypeError(msg)


def _convert_dict_to_message(message_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a Maritaca message dict to LangChain message.

    Args:
        message_dict: Dictionary from Maritaca API response.

    Returns:
        LangChain BaseMessage.
    """
    role = message_dict.get("role", "")
    content = message_dict.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return ChatMessage(content=content, role=role)


def _create_usage_metadata(token_usage: dict[str, Any]) -> UsageMetadata:
    """Create usage metadata from Maritaca token usage response.

    Args:
        token_usage: Token usage dict from Maritaca API response.

    Returns:
        UsageMetadata with token counts.
    """
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
