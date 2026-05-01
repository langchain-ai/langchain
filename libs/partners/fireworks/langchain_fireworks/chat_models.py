"""Fireworks chat wrapper."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import (
    Any,
    Literal,
    NoReturn,
    cast,
)

import httpx
from fireworks.client import AsyncFireworks, Fireworks  # type: ignore[import-untyped]
from fireworks.client.error import (  # type: ignore[import-untyped]
    APITimeoutError,
    BadGatewayError,
    FireworksError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
    is_data_content_block,
)
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)
from langchain_core.messages.tool import (
    ToolCallChunk,
)
from langchain_core.messages.tool import (
    tool_call_chunk as create_tool_call_chunk,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

from langchain_fireworks._compat import _convert_from_v1_to_chat_completions
from langchain_fireworks.data._profiles import _PROFILES

logger = logging.getLogger(__name__)


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.

    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        # Fix for azure
        # Also Fireworks returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if reasoning_content := _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = reasoning_content

        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)

        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=_dict.get("name", "")
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
            additional_kwargs=additional_kwargs,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role or "")


def _format_message_content(content: Any) -> Any:
    """Format message content for the Fireworks chat completions wire format.

    Adapted from `langchain_openai.chat_models.base._format_message_content`,
    scoped to the chat completions API: drops content block types the wire
    format does not carry, translates canonical v0/v1 multimodal data blocks
    via `convert_to_openai_data_block(block, api="chat/completions")`, and
    converts legacy Anthropic-shape image blocks (`{"type": "image",
    "source": {...}}`) to OpenAI `image_url` blocks. String and non-list
    content are returned unchanged.

    Args:
        content: The message content. Strings and non-list values are
            returned as-is; lists are walked block by block.

    Returns:
        The formatted content, ready to be placed on the chat completions
        wire. List inputs return a new list with translations applied; other
        inputs are returned unchanged.
    """
    if not isinstance(content, list):
        return content
    formatted: list[Any] = []
    for block in content:
        if isinstance(block, dict) and "type" in block:
            btype = block["type"]
            if btype in (
                "tool_use",
                "thinking",
                "reasoning_content",
                "function_call",
                "code_interpreter_call",
            ):
                continue
            if is_data_content_block(block):
                formatted.append(
                    convert_to_openai_data_block(block, api="chat/completions")
                )
                continue
            if (
                btype == "image"
                and (source := block.get("source"))
                and isinstance(source, dict)
            ):
                if (
                    source.get("type") == "base64"
                    and (media_type := source.get("media_type"))
                    and (data := source.get("data"))
                ):
                    formatted.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
                    continue
                if source.get("type") == "url" and (url := source.get("url")):
                    formatted.append({"type": "image_url", "image_url": {"url": url}})
                    continue
                continue
        formatted.append(block)
    return formatted


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.

    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {
            "role": message.role,
            "content": _format_message_content(message.content),
        }
    elif isinstance(message, HumanMessage):
        message_dict = {
            "role": "user",
            "content": _format_message_content(message.content),
        }
    elif isinstance(message, AIMessage):
        # Translate v1 content
        if message.response_metadata.get("output_version") == "v1":
            message = _convert_from_v1_to_chat_completions(message)
        message_dict = {
            "role": "assistant",
            "content": _format_message_content(message.content),
        }
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_fireworks_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_fireworks_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        # If tool calls only, content is None not empty string
        if "tool_calls" in message_dict and message_dict["content"] == "":
            message_dict["content"] = None
        else:
            pass
    elif isinstance(message, SystemMessage):
        message_dict = {
            "role": "system",
            "content": _format_message_content(message.content),
        }
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": _format_message_content(message.content),
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _usage_to_metadata(usage: Mapping[str, Any]) -> dict[str, int]:
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
    }


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choices = chunk.get("choices") or []
    response_metadata: dict[str, Any] = {"model_provider": "fireworks"}
    if service_tier := chunk.get("service_tier"):
        response_metadata["service_tier"] = service_tier
    if not choices:
        # Final chunk emitted when `stream_options.include_usage=True`:
        # `choices` is empty and the chunk carries only `usage`.
        usage = chunk.get("usage")
        if not usage:
            logger.debug(
                "Received stream chunk with no choices and no usage: %s", chunk
            )
        usage_metadata = _usage_to_metadata(usage) if usage else None
        return AIMessageChunk(
            content="",
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
            response_metadata=response_metadata,
        )
    choice = choices[0]
    _dict = choice["delta"]
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    tool_call_chunks: list[ToolCallChunk] = []
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for rtc in raw_tool_calls:
            with contextlib.suppress(KeyError):
                tool_call_chunks.append(
                    create_tool_call_chunk(
                        name=rtc["function"].get("name"),
                        args=rtc["function"].get("arguments"),
                        id=rtc.get("id"),
                        index=rtc.get("index"),
                    )
                )
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        usage = chunk.get("usage")
        usage_metadata = _usage_to_metadata(usage) if usage else None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
            response_metadata=response_metadata,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


class _RetryableHTTPStatusError(FireworksError):
    """Internal marker for 5xx `httpx.HTTPStatusError` responses.

    The Fireworks SDK maps a subset of status codes (500, 502, 503) to typed
    exceptions but lets others (504, 507-511, Cloudflare-edge 520-599)
    propagate as raw `httpx.HTTPStatusError`. Promoting those to this marker
    inside `_call` keeps the retryable set expressible as a list of classes
    for `create_base_retry_decorator`, preserving parity with `ChatMistralAI`.
    """


_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    APITimeoutError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    httpx.TimeoutException,
    httpx.TransportError,
    _RetryableHTTPStatusError,
)


def _promote_http_status_error(exc: httpx.HTTPStatusError) -> NoReturn:
    """Re-raise 5xx `httpx.HTTPStatusError` as a retryable marker."""
    if exc.response.status_code >= 500:
        msg = f"Retryable {exc.response.status_code} from Fireworks: {exc}"
        raise _RetryableHTTPStatusError(msg) from exc
    raise exc


def _raise_empty_stream() -> NoReturn:
    """Raise a descriptive error when the SDK returns a zero-chunk stream."""
    msg = "Received empty stream from Fireworks"
    raise FireworksError(msg)


def _create_retry_decorator(
    llm: ChatFireworks,
    run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None = None,
) -> Callable[[Any], Any]:
    """Return a tenacity retry decorator for Fireworks SDK calls.

    Retries are implemented here because the pinned Fireworks SDK 0.x does
    not honor its own `_max_retries` attribute on completion resources.
    """
    # `max_retries` counts retries *after* the initial attempt.
    # `create_base_retry_decorator` forwards its `max_retries` to
    # `stop_after_attempt`, which counts total attempts — so offset by 1.
    # Note: this diverges from `ChatMistralAI`, which passes the raw value;
    # the fireworks field docstring is the source of truth here.
    # `None` and `0` both mean "single attempt, no retries".
    attempts = (llm.max_retries + 1) if llm.max_retries else 1
    return create_base_retry_decorator(
        error_types=list(_RETRYABLE_ERRORS),
        max_retries=attempts,
        run_manager=run_manager,
    )


def _completion_with_retry(
    llm: ChatFireworks,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Any:
    """Retry the sync completion call, including stream setup."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _call() -> Any:
        try:
            result = llm.client.create(**kwargs)
        except httpx.HTTPStatusError as e:
            _promote_http_status_error(e)
        if kwargs.get("stream"):
            # The streaming generator is lazy — advance once so the HTTP
            # connection and any transport error happen inside the retry
            # boundary. `_prepend_chunk` then re-yields the consumed chunk
            # ahead of the rest so callers still see every event.
            try:
                iterator = iter(result)
                first = next(iterator)
            except StopIteration:
                _raise_empty_stream()
            except httpx.HTTPStatusError as e:
                _promote_http_status_error(e)
            return _prepend_chunk(first, iterator)
        return result

    return _call()


async def _acompletion_with_retry(
    llm: ChatFireworks,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Any:
    """Retry the async completion call, including stream setup."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _call() -> Any:
        if kwargs.get("stream"):
            try:
                result = llm.async_client.acreate(**kwargs)
                agen = result.__aiter__()
                first = await agen.__anext__()
            except StopAsyncIteration:
                _raise_empty_stream()
            except httpx.HTTPStatusError as e:
                _promote_http_status_error(e)
            return _aprepend_chunk(first, agen)
        try:
            return await llm.async_client.acreate(**kwargs)
        except httpx.HTTPStatusError as e:
            _promote_http_status_error(e)

    return await _call()


def _prepend_chunk(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from rest


async def _aprepend_chunk(first: Any, rest: AsyncIterator[Any]) -> AsyncIterator[Any]:
    yield first
    async for item in rest:
        yield item


# This is basically a copy and replace for ChatFireworks, except
# - I needed to gut out tiktoken and some of the token estimation logic
# (not sure how important it is)
# - Environment variable is different
# we should refactor into some OpenAI-like class in the future
class ChatFireworks(BaseChatModel):
    """`Fireworks` Chat large language models API.

    To use, you should have the
    environment variable `FIREWORKS_API_KEY` set with your API key.

    Any parameters that are valid to be passed to the fireworks.create call
    can be passed in, even if not explicitly saved on this class.

    Example:
        ```python
        from langchain_fireworks.chat_models import ChatFireworks

        fireworks = ChatFireworks(model_name="accounts/fireworks/models/gpt-oss-120b")
        ```
    """

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"fireworks_api_key": "FIREWORKS_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "chat_models", "fireworks"]`
        """
        return ["langchain", "chat_models", "fireworks"]

    @property
    def lc_attributes(self) -> dict[str, Any]:
        attributes: dict[str, Any] = {}
        if self.fireworks_api_base:
            attributes["fireworks_api_base"] = self.fireworks_api_base

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    client: Any = Field(default=None, exclude=True)

    async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(alias="model")
    """Model name to use."""

    @property
    def model(self) -> str:
        """Same as model_name."""
        return self.model_name

    temperature: float | None = None
    """What sampling temperature to use."""

    stop: str | list[str] | None = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    fireworks_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "FIREWORKS_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `FIREWORKS_API_KEY`."
            ),
        ),
    )
    """Fireworks API key.

    Automatically read from env variable `FIREWORKS_API_KEY` if not provided.
    """

    fireworks_api_base: str | None = Field(
        alias="base_url", default_factory=from_env("FIREWORKS_API_BASE", default=None)
    )
    """Base URL path for API requests, leave blank if not using a proxy or service
    emulator.
    """

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Fireworks completion API. Can be `float`,
    `httpx.Timeout` or `None`.
    """

    streaming: bool = False
    """Whether to stream the results or not."""

    stream_usage: bool = True
    """Whether to include usage metadata in streaming output.

    If `True`, a final empty-content chunk carrying `usage_metadata` is emitted
    during the stream. Set to `False` if the upstream model/proxy rejects
    `stream_options`, or pass `stream_options` explicitly via `model_kwargs` or
    a runtime kwarg to override.

    !!! version-added "Added in `langchain-fireworks` 1.2.0"

    !!! warning "Behavior changed in `langchain-fireworks` 1.2.0"

        Streaming now opts into `stream_options.include_usage` by default, and
        the final empty-`choices` chunk is surfaced as an `AIMessageChunk` with
        `usage_metadata` instead of being silently dropped.
    """

    n: int = 1
    """Number of chat completions to generate for each prompt."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    max_retries: int | None = None
    """Maximum number of retries after the initial attempt when generating.

    Retries use exponential backoff and trigger on transient errors:
    `RateLimitError`, `APITimeoutError`, 5xx responses (including those that
    surface as `httpx.HTTPStatusError` rather than typed SDK errors), and
    underlying transport errors (`httpx.TimeoutException`, `httpx.TransportError`).
    A value of `None` or `0` disables retries.
    """

    service_tier: str | None = None
    """Service tier for the request.

    Forwarded as the `service_tier` field on the Fireworks chat completions
    request when set. Pass `'priority'` to opt into Fireworks' priority tier;
    leave as `None` to use the default tier.

    To use Fireworks' fast mode instead, select a fast-routed `model`; fast mode
    is not controlled by this field. See Fireworks'
    [serverless product docs](https://docs.fireworks.ai/guides/serverless-products)
    for the current list of fast routers and tiers.

    !!! version-added "Added in `langchain-fireworks` 1.3.0"
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        client_params = {
            "api_key": (
                self.fireworks_api_key.get_secret_value()
                if self.fireworks_api_key
                else None
            ),
            "base_url": self.fireworks_api_base,
            "timeout": self.request_timeout,
        }

        if not self.client:
            self.client = Fireworks(**client_params).chat.completions
        if not self.async_client:
            self.async_client = AsyncFireworks(**client_params).chat.completions
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling Fireworks API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "n": self.n,
            "stop": self.stop,
            **self.model_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.service_tier is not None:
            params["service_tier"] = self.service_tier
        return params

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="fireworks",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage and "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in _completion_with_retry(
            self, run_manager=run_manager, messages=message_dicts, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info: dict[str, Any] = {}
            logprobs = None
            if choices := chunk.get("choices"):
                choice = choices[0]
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                    generation_info["model_name"] = self.model_name
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            yield generation_chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = _completion_with_retry(
            self, run_manager=run_manager, messages=message_dicts, **params
        )
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: dict | BaseModel) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        token_usage = response.get("usage", {})
        service_tier = response.get("service_tier")
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if isinstance(message, AIMessage):
                if token_usage:
                    message.usage_metadata = {
                        "input_tokens": token_usage.get("prompt_tokens", 0),
                        "output_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("total_tokens", 0),
                    }
                    message.response_metadata["model_provider"] = "fireworks"
                    message.response_metadata["model_name"] = self.model_name
                if service_tier:
                    message.response_metadata["service_tier"] = service_tier
            generation_info = {"finish_reason": res.get("finish_reason")}
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        if service_tier:
            llm_output["service_tier"] = service_tier
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage and "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in await _acompletion_with_retry(
            self, run_manager=run_manager, messages=message_dicts, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info: dict[str, Any] = {}
            logprobs = None
            if choices := chunk.get("choices"):
                choice = choices[0]
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                    generation_info["model_name"] = self.model_name
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = await _acompletion_with_retry(
            self, run_manager=run_manager, messages=message_dicts, **params
        )
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}

    def _get_invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "fireworks-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with Fireworks tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.

                Supports any tool definition handled by [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool].
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                `'auto'` to automatically determine which function to call
                with the option to not call any function, `'any'` to enforce that some
                function is called, or a dict of the form:
                `{"type": "function", "function": {"name": <<tool_name>>}}`.
            **kwargs: Any additional parameters to pass to
                `langchain_fireworks.chat_models.ChatFireworks.bind`
        """  # noqa: E501
        strict = kwargs.pop("strict", None)
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "any", "none")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    msg = (
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                    raise ValueError(msg)
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type[BaseModel] | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - An OpenAI function/tool schema,
                - A JSON Schema,
                - A `TypedDict` class,
                - Or a Pydantic class.

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.

            method: The method for steering model generation, one of:

                - `'function_calling'`:
                    Uses Fireworks's [tool-calling features](https://docs.fireworks.ai/guides/function-calling).
                - `'json_schema'`:
                    Uses Fireworks's [structured output feature](https://docs.fireworks.ai/structured-responses/structured-response-formatting).
                - `'json_mode'`:
                    Uses Fireworks's [JSON mode feature](https://docs.fireworks.ai/structured-responses/structured-response-formatting).

                !!! warning "Behavior changed in `langchain-fireworks` 0.2.8"

                    Added support for `'json_schema'`.

            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

            kwargs:
                Any additional parameters to pass to the `langchain.runnable.Runnable`
                constructor.

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        Example: schema=Pydantic class, method="function_calling", include_raw=False:

        ```python
        from typing import Optional

        from langchain_fireworks import ChatFireworks
        from pydantic import BaseModel, Field


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            # If we provide default values and/or descriptions for fields, these will be passed
            # to the model. This is an important part of improving a model's ability to
            # correctly return structured outputs.
            justification: str | None = Field(
                default=None, description="A justification for the answer."
            )


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(AnswerWithJustification)

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )

        # -> AnswerWithJustification(
        #     answer='They weigh the same',
        #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
        # )
        ```

        Example: schema=Pydantic class, method="function_calling", include_raw=True:

        ```python
        from langchain_fireworks import ChatFireworks
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: str


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(
            AnswerWithJustification, include_raw=True
        )

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )
        # -> {
        #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
        #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
        #     'parsing_error': None
        # }
        ```

        Example: schema=TypedDict class, method="function_calling", include_raw=False:

        ```python
        from typing_extensions import Annotated, TypedDict

        from langchain_fireworks import ChatFireworks


        class AnswerWithJustification(TypedDict):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: Annotated[
                str | None, None, "A justification for the answer."
            ]


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(AnswerWithJustification)

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )
        # -> {
        #     'answer': 'They weigh the same',
        #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
        # }
        ```

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:

        ```python
        from langchain_fireworks import ChatFireworks

        oai_schema = {
            "name": "AnswerWithJustification",
            "description": "An answer to the user question along with justification for the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "justification": {
                        "description": "A justification for the answer.",
                        "type": "string",
                    },
                },
                "required": ["answer"],
            },
        }

        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(oai_schema)

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )
        # -> {
        #     'answer': 'They weigh the same',
        #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
        # }
        ```

        Example: schema=Pydantic class, method="json_mode", include_raw=True:

        ```python
        from langchain_fireworks import ChatFireworks
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            answer: str
            justification: str


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b", temperature=0
        )
        structured_model = model.with_structured_output(
            AnswerWithJustification, method="json_mode", include_raw=True
        )

        structured_model.invoke(
            "Answer the following question. "
            "Make sure to return a JSON blob with keys 'answer' and 'justification'. "
            "What's heavier a pound of bricks or a pound of feathers?"
        )
        # -> {
        #     'raw': AIMessage(content='{"answer": "They are both the same weight.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight."}'),
        #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
        #     'parsing_error': None
        # }
        ```

        Example: schema=None, method="json_mode", include_raw=True:

        ```python
        structured_model = model.with_structured_output(
            method="json_mode", include_raw=True
        )

        structured_model.invoke(
            "Answer the following question. "
            "Make sure to return a JSON blob with keys 'answer' and 'justification'. "
            "What's heavier a pound of bricks or a pound of feathers?"
        )
        # -> {
        #     'raw': AIMessage(content='{"answer": "They are both the same weight.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight."}'),
        #     'parsed': {
        #         'answer': 'They are both the same weight.',
        #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
        #     },
        #     'parsing_error': None
        # }
        ```

        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_schema = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={"type": "json_object", "schema": formatted_schema},
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _lc_tool_call_to_fireworks_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_fireworks_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }
