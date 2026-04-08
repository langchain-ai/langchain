"""Sarvam AI Chat wrapper."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
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
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_sarvam._version import __version__


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to the Sarvam API dict format."""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg_dict: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        return msg_dict
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    elif isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content}
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")


def _create_usage_metadata(usage: Any) -> UsageMetadata:
    """Build UsageMetadata from a Sarvam usage response object."""
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens) or 0
    else:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = (
            getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


class ChatSarvam(BaseChatModel):
    r"""Sarvam AI chat large language model.

    Sarvam AI offers multilingual chat models with native support for Indian
    languages and advanced reasoning capabilities.

    Setup:
        Install ``langchain-sarvam`` and set the ``SARVAM_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install -U langchain-sarvam
            export SARVAM_API_KEY="your-api-key"

    Key init args — completion params:
        model:
            Name of the Sarvam model to use.
            Options: ``"sarvam-m"`` (legacy 24B), ``"sarvam-30b"``,
            ``"sarvam-105b"``.
        temperature:
            Sampling temperature between 0 and 2. Default 0.7.
        max_tokens:
            Maximum number of tokens to generate.
        reasoning_effort:
            Controls the depth of chain-of-thought reasoning.
            One of ``"low"``, ``"medium"``, ``"high"``.

    Key init args — client params:
        api_key:
            Sarvam API key. Reads from ``SARVAM_API_KEY`` env var if not
            provided.
        base_url:
            Custom base URL for the Sarvam API. Reads from
            ``SARVAM_API_BASE`` env var if not provided.

    Instantiate:
        .. code-block:: python

            from langchain_sarvam import ChatSarvam

            model = ChatSarvam(
                model="sarvam-m",
                temperature=0.7,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "Translate 'Good morning' to Hindi."),
            ]
            model.invoke(messages)

        .. code-block:: python

            AIMessage(content='सुप्रभात (Suprabhat)', ...)

    Stream:
        .. code-block:: python

            for chunk in model.stream(messages):
                print(chunk.text, end="")

    Async:
        .. code-block:: python

            await model.ainvoke(messages)

    Response metadata:
        .. code-block:: python

            ai_msg = model.invoke(messages)
            ai_msg.response_metadata
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(default="sarvam-m", alias="model")
    """Sarvam model name. Options: ``sarvam-m``, ``sarvam-30b``, ``sarvam-105b``."""

    temperature: float = 0.7
    """Sampling temperature between 0 and 2."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    top_p: float | None = None
    """Nucleus sampling parameter between 0 and 1."""

    n: int = 1
    """Number of completions to generate for each prompt."""

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """Stop sequences."""

    streaming: bool = False
    """Whether to stream responses."""

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    """Reasoning depth for complex tasks. One of ``low``, ``medium``, ``high``."""

    frequency_penalty: float | None = None
    """Penalises repeated tokens (-2.0 to 2.0)."""

    presence_penalty: float | None = None
    """Penalises new topics (-2.0 to 2.0)."""

    seed: int | None = None
    """Optional seed for deterministic sampling (best-effort)."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra keyword arguments forwarded verbatim to the Sarvam API."""

    sarvam_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("SARVAM_API_KEY", default=None),
    )
    """Sarvam API key. Falls back to ``SARVAM_API_KEY`` environment variable."""

    sarvam_api_base: str | None = Field(
        alias="base_url",
        default_factory=from_env("SARVAM_API_BASE", default=None),
    )
    """Optional custom base URL for the Sarvam API endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that the ``sarvamai`` package is installed and build clients."""
        try:
            from sarvamai import AsyncSarvamAI, SarvamAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            ) from exc

        api_key = (
            self.sarvam_api_key.get_secret_value() if self.sarvam_api_key else None
        )

        client_kwargs: dict[str, Any] = {"api_subscription_key": api_key}
        if self.sarvam_api_base:
            client_kwargs["base_url"] = self.sarvam_api_base

        if not self.client:
            self.client = SarvamAI(**client_kwargs)
        if not self.async_client:
            self.async_client = AsyncSarvamAI(**client_kwargs)

        return self

    # ------------------------------------------------------------------
    # LangChain serialisation helpers
    # ------------------------------------------------------------------

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"sarvam_api_key": "SARVAM_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _llm_type(self) -> str:
        return "sarvam-chat"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _default_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "n": self.n,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop is not None:
            params["stop"] = self.stop
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            params["seed"] = self.seed
        return params

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params.copy()
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Convert a Sarvam chat completion response to a LangChain ChatResult."""
        if not isinstance(response, dict):
            response = response.model_dump() if hasattr(response, "model_dump") else vars(response)

        generations: list[ChatGeneration] = []
        token_usage = response.get("usage") or {}

        for choice in response.get("choices", []):
            msg_data = choice.get("message", {})
            content = msg_data.get("content") or ""
            ai_message = AIMessage(content=content)

            if token_usage:
                ai_message.usage_metadata = _create_usage_metadata(token_usage)

            gen_info: dict[str, Any] = {
                "finish_reason": choice.get("finish_reason"),
                "model_name": response.get("model", self.model_name),
            }
            if generation_info:
                gen_info.update(generation_info)

            generations.append(
                ChatGeneration(message=ai_message, generation_info=gen_info)
            )

        llm_output: dict[str, Any] = {
            "token_usage": token_usage,
            "model_name": response.get("model", self.model_name),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    # ------------------------------------------------------------------
    # Sync generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params.update(kwargs)
        response = self.client.chat.completions(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params.update(kwargs)
        params["stream"] = True

        response = self.client.chat.completions(messages=message_dicts, **params)

        for chunk in self._iter_sse_chunks(response):
            if not chunk:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content") or ""
            finish_reason = choice.get("finish_reason")

            gen_info: dict[str, Any] = {}
            if finish_reason:
                gen_info["finish_reason"] = finish_reason

            message_chunk = AIMessageChunk(content=content)
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=gen_info or None,
            )

            if run_manager:
                run_manager.on_llm_new_token(content, chunk=generation_chunk)
            yield generation_chunk

    # ------------------------------------------------------------------
    # Async generation
    # ------------------------------------------------------------------

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params.update(kwargs)
        response = await self.async_client.chat.completions(
            messages=message_dicts, **params
        )
        return self._create_chat_result(response)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params.update(kwargs)
        params["stream"] = True

        response = await self.async_client.chat.completions(
            messages=message_dicts, **params
        )

        async for chunk in self._aiter_sse_chunks(response):
            if not chunk:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content") or ""
            finish_reason = choice.get("finish_reason")

            gen_info: dict[str, Any] = {}
            if finish_reason:
                gen_info["finish_reason"] = finish_reason

            message_chunk = AIMessageChunk(content=content)
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=gen_info or None,
            )

            if run_manager:
                await run_manager.on_llm_new_token(content, chunk=generation_chunk)
            yield generation_chunk

    # ------------------------------------------------------------------
    # SSE chunk parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_sse_chunks(response: Any) -> Iterator[dict[str, Any]]:
        """Iterate over streaming chunks from the Sarvam SDK.

        The sarvamai SDK may return either an iterable of parsed objects or
        raw server-sent-event strings. Both cases are handled here.
        """
        for raw in response:
            if isinstance(raw, dict):
                yield raw
            elif hasattr(raw, "model_dump"):
                yield raw.model_dump()
            elif isinstance(raw, (bytes, str)):
                line = raw.decode() if isinstance(raw, bytes) else raw
                line = line.strip()
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    if data and data != "[DONE]":
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            pass

    @staticmethod
    async def _aiter_sse_chunks(response: Any) -> AsyncIterator[dict[str, Any]]:
        """Async version of :meth:`_iter_sse_chunks`."""
        async for raw in response:
            if isinstance(raw, dict):
                yield raw
            elif hasattr(raw, "model_dump"):
                yield raw.model_dump()
            elif isinstance(raw, (bytes, str)):
                line = raw.decode() if isinstance(raw, bytes) else raw
                line = line.strip()
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    if data and data != "[DONE]":
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            pass
