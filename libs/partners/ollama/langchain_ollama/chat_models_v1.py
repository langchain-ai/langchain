"""Ollama chat model v1 implementation.

This implementation provides native support for v1 messages with structured
content blocks and always returns AIMessageV1 format responses.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Callable, Literal, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.language_models.v1.chat_models import BaseChatModelV1
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.v1 import AIMessage as AIMessageV1
from langchain_core.messages.v1 import AIMessageChunk as AIMessageChunkV1
from langchain_core.messages.v1 import MessageV1
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from ollama import AsyncClient, Client, Options
from pydantic import PrivateAttr, model_validator
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import Self

from ._compat import (
    _convert_chunk_to_v1,
    _convert_from_v1_to_ollama_format,
    _convert_to_v1_from_ollama_format,
)
from ._utils import validate_model

log = logging.getLogger(__name__)


def _get_usage_metadata_from_response(
    response: dict[str, Any],
) -> Optional[UsageMetadata]:
    """Extract usage metadata from Ollama response."""
    input_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")
    if input_tokens is not None and output_tokens is not None:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    return None


class BaseChatOllamaV1(BaseChatModelV1):
    """Base class for Ollama v1 chat models."""

    model: str
    """Model name to use."""

    reasoning: Optional[bool] = None
    """Controls the reasoning/thinking mode for supported models.

    - ``True``: Enables reasoning mode. The model's reasoning process will be
      captured and returned as a ``ReasoningContentBlock`` in the response
      message content. The main response content will not include the reasoning tags.
    - ``False``: Disables reasoning mode. The model will not perform any reasoning,
      and the response will not include any reasoning content.
    - ``None`` (Default): The model will use its default reasoning behavior. Note
      however, if the model's default behavior *is* to perform reasoning, think tags
      (``<think>`` and ``</think>``) will be present within the main response content
      unless you set ``reasoning`` to ``True``.
    """

    validate_model_on_init: bool = False
    """Whether to validate the model exists in Ollama locally on initialization."""

    # Ollama-specific parameters
    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity."""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback."""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity."""

    num_ctx: Optional[int] = None
    """Sets the size of the context window."""

    num_gpu: Optional[int] = None
    """The number of GPUs to use."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation."""

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict."""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent repetition."""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions."""

    temperature: Optional[float] = None
    """The temperature of the model."""

    seed: Optional[int] = None
    """Sets the random number seed to use for generation."""

    stop: Optional[list[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling parameter."""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense."""

    top_p: Optional[float] = None
    """Works together with top-k."""

    format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None
    """Specify the format of the output."""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx clients."""

    async_client_kwargs: Optional[dict] = {}
    """Additional kwargs for the async httpx client."""

    sync_client_kwargs: Optional[dict] = {}
    """Additional kwargs for the sync httpx client."""

    _client: Client = PrivateAttr()
    _async_client: AsyncClient = PrivateAttr()

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients to use for ollama."""
        client_kwargs = self.client_kwargs or {}

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=self.base_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **async_client_kwargs)
        if self.validate_model_on_init:
            validate_model(self._client, self.model)
        return self

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="ollama",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _get_invocation_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Get parameters for model invocation."""
        params = {
            "model": self.model,
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "num_predict": self.num_predict,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "seed": self.seed,
            "stop": stop or self.stop,
            "tfs_z": self.tfs_z,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "format": self.format,
            "keep_alive": self.keep_alive,
        }
        params.update(kwargs)
        return params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ollama-v1"


class ChatOllamaV1(BaseChatOllamaV1):
    """Ollama chat model with native v1 content block support.

    This implementation provides native support for structured content blocks
    and always returns AIMessageV1 format responses.

    Examples:
        Basic text conversation:

        .. code-block:: python

            from langchain_ollama import ChatOllamaV1
            from langchain_core.messages.v1 import HumanMessage
            from langchain_core.messages.content_blocks import TextContentBlock

            llm = ChatOllamaV1(model="llama3")
            response = llm.invoke([
                HumanMessage(content=[
                    TextContentBlock(type="text", text="Hello!")
                ])
            ])

            # Response is always structured
            print(response.content)
            # [{"type": "text", "text": "Hello! How can I help?"}]

        Multi-modal input:

        .. code-block:: python

            from langchain_core.messages.content_blocks import ImageContentBlock

            response = llm.invoke([
                HumanMessage(content=[
                    TextContentBlock(type="text", text="Describe this image:"),
                    ImageContentBlock(
                        type="image",
                        mime_type="image/jpeg",
                        data="base64_encoded_image",
                        source_type="base64"
                    )
                ])
            ])
    """

    def _chat_params(
        self,
        messages: list[MessageV1],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build parameters for Ollama chat API."""
        # Convert v1 messages to Ollama format
        ollama_messages = [_convert_from_v1_to_ollama_format(msg) for msg in messages]

        if self.stop is not None and stop is not None:
            msg = "`stop` found in both the input and default params."
            raise ValueError(msg)
        if self.stop is not None:
            stop = self.stop

        options_dict = kwargs.pop(
            "options",
            {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "seed": self.seed,
                "stop": self.stop if stop is None else stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        )

        params = {
            "messages": ollama_messages,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "think": kwargs.pop("reasoning", self.reasoning),
            "format": kwargs.pop("format", self.format),
            "options": Options(**options_dict),
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

        if tools := kwargs.get("tools"):
            params["tools"] = tools

        return params

    def _generate_stream(
        self,
        messages: list[MessageV1],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunkV1]:
        """Generate streaming response with native v1 chunks."""
        chat_params = self._chat_params(messages, stop, **kwargs)

        if chat_params["stream"]:
            for part in self._client.chat(**chat_params):
                if not isinstance(part, str):
                    # Skip empty load responses
                    if (
                        part.get("done") is True
                        and part.get("done_reason") == "load"
                        and not part.get("message", {}).get("content", "").strip()
                    ):
                        log.warning(
                            "Ollama returned empty response with done_reason='load'. "
                            "Skipping this response."
                        )
                        continue

                    chunk = _convert_chunk_to_v1(part)

                    # Add usage metadata for final chunks
                    if part.get("done") is True:
                        usage_metadata = _get_usage_metadata_from_response(part)
                        if usage_metadata:
                            chunk.usage_metadata = usage_metadata

                    if run_manager:
                        text_content = "".join(
                            str(block.get("text", ""))
                            for block in chunk.content
                            if block.get("type") == "text"
                        )
                        run_manager.on_llm_new_token(
                            text_content,
                            chunk=chunk,
                        )
                    yield chunk
        else:
            # Non-streaming case
            response = self._client.chat(**chat_params)
            ai_message = _convert_to_v1_from_ollama_format(response)
            usage_metadata = _get_usage_metadata_from_response(response)
            if usage_metadata:
                ai_message.usage_metadata = usage_metadata
            # Convert to chunk for yielding
            chunk = AIMessageChunkV1(
                content=ai_message.content,
                response_metadata=ai_message.response_metadata,
                usage_metadata=ai_message.usage_metadata,
            )
            yield chunk

    async def _agenerate_stream(
        self,
        messages: list[MessageV1],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunkV1]:
        """Generate async streaming response with native v1 chunks."""
        chat_params = self._chat_params(messages, stop, **kwargs)

        if chat_params["stream"]:
            async for part in await self._async_client.chat(**chat_params):
                if not isinstance(part, str):
                    # Skip empty load responses
                    if (
                        part.get("done") is True
                        and part.get("done_reason") == "load"
                        and not part.get("message", {}).get("content", "").strip()
                    ):
                        log.warning(
                            "Ollama returned empty response with done_reason='load'. "
                            "Skipping this response."
                        )
                        continue

                    chunk = _convert_chunk_to_v1(part)

                    # Add usage metadata for final chunks
                    if part.get("done") is True:
                        usage_metadata = _get_usage_metadata_from_response(part)
                        if usage_metadata:
                            chunk.usage_metadata = usage_metadata

                    if run_manager:
                        text_content = "".join(
                            str(block.get("text", ""))
                            for block in chunk.content
                            if block.get("type") == "text"
                        )
                        await run_manager.on_llm_new_token(
                            text_content,
                            chunk=chunk,
                        )
                    yield chunk
        else:
            # Non-streaming case
            response = await self._async_client.chat(**chat_params)
            ai_message = _convert_to_v1_from_ollama_format(response)
            usage_metadata = _get_usage_metadata_from_response(response)
            if usage_metadata:
                ai_message.usage_metadata = usage_metadata
            # Convert to chunk for yielding
            chunk = AIMessageChunkV1(
                content=ai_message.content,
                response_metadata=ai_message.response_metadata,
                usage_metadata=ai_message.usage_metadata,
            )
            yield chunk

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessageV1]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
            tool_choice: Tool choice parameter (currently ignored by Ollama).
            kwargs: Additional parameters passed to bind().
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
