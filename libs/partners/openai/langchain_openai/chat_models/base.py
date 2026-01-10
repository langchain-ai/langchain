"""OpenAI chat wrapper."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import ssl
import sys
import warnings
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from functools import partial
from io import BytesIO
from json import JSONDecodeError
from math import ceil
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)
from urllib.parse import urlparse

import certifi
import openai
import tiktoken
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
)
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
from langchain_core.messages import content as types
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.block_translators.openai import (
    _convert_from_v03_ai_message,
    convert_to_openai_data_block,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool
from langchain_core.tools.base import _stringify
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    PydanticBaseModel,
    TypeBaseModel,
    is_basemodel_subclass,
)
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self

from langchain_openai.chat_models._client_utils import (
    _get_default_async_httpx_client,
    _get_default_httpx_client,
    _resolve_sync_and_async_api_keys,
)
from langchain_openai.chat_models._compat import (
    _convert_from_v1_to_chat_completions,
    _convert_from_v1_to_responses,
    _convert_to_v03_ai_message,
)
from langchain_openai.data._profiles import _PROFILES

if TYPE_CHECKING:
    from langchain_core.language_models import ModelProfile
    from openai.types.responses import Response

logger = logging.getLogger(__name__)

# This SSL context is equivelent to the default `verify=True`.
# https://www.python-httpx.org/advanced/ssl/#configuring-client-instances
global_ssl_context = ssl.create_default_context(cafile=certifi.where())

_MODEL_PROFILES = cast(ModelProfileRegistry, _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


WellKnownTools = (
    "file_search",
    "web_search_preview",
    "web_search",
    "computer_use_preview",
    "code_interpreter",
    "mcp",
    "image_generation",
)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    if role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(
            content=_dict.get("content", ""),
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[arg-type]


def _format_message_content(
    content: Any,
    api: Literal["chat/completions", "responses"] = "chat/completions",
    role: str | None = None,
) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        formatted_content = []
        for block in content:
            # Remove unexpected block types
            if (
                isinstance(block, dict)
                and "type" in block
                and (
                    block["type"] in ("tool_use", "thinking", "reasoning_content")
                    or (
                        block["type"] in ("function_call", "code_interpreter_call")
                        and api == "chat/completions"
                    )
                )
            ):
                continue
            if (
                isinstance(block, dict)
                and is_data_content_block(block)
                # Responses API messages handled separately in _compat (parsed into
                # image generation calls)
                and not (api == "responses" and str(role).lower().startswith("ai"))
            ):
                formatted_content.append(convert_to_openai_data_block(block, api=api))
            # Anthropic image blocks
            elif (
                isinstance(block, dict)
                and block.get("type") == "image"
                and (source := block.get("source"))
                and isinstance(source, dict)
            ):
                if source.get("type") == "base64" and (
                    (media_type := source.get("media_type"))
                    and (data := source.get("data"))
                ):
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
                elif source.get("type") == "url" and (url := source.get("url")):
                    formatted_content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                else:
                    continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _convert_message_to_dict(
    message: BaseMessage,
    api: Literal["chat/completions", "responses"] = "chat/completions",
) -> dict:
    """Convert a LangChain message to dictionary format expected by OpenAI."""
    message_dict: dict[str, Any] = {
        "content": _format_message_content(message.content, api=api, role=message.type)
    }
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        elif "function_call" in message.additional_kwargs:
            # OpenAI raises 400 if both function_call and tool_calls are present in the
            # same message.
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        audio: dict[str, Any] | None = None
        for block in message.content:
            if (
                isinstance(block, dict)
                and block.get("type") == "audio"
                and (id_ := block.get("id"))
                and api != "responses"
            ):
                # openai doesn't support passing the data back - only the id
                # https://platform.openai.com/docs/guides/audio/multi-turn-conversations
                audio = {"id": id_}
        if not audio and "audio" in message.additional_kwargs:
            raw_audio = message.additional_kwargs["audio"]
            audio = (
                {"id": message.additional_kwargs["audio"]["id"]}
                if "id" in raw_audio
                else raw_audio
            )
        if audio:
            message_dict["audio"] = audio
    elif isinstance(message, SystemMessage):
        message_dict["role"] = message.additional_kwargs.get(
            "__openai_role__", "system"
        )
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert to a LangChain message chunk."""
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        else:
            additional_kwargs = {}
        return SystemMessageChunk(
            content=content, id=id_, additional_kwargs=additional_kwargs
        )
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


def _update_token_usage(
    overall_token_usage: int | dict, new_usage: int | dict
) -> int | dict:
    # Token usage is either ints or dictionaries
    # `reasoning_tokens` is nested inside `completion_tokens_details`
    if isinstance(new_usage, int):
        if not isinstance(overall_token_usage, int):
            msg = (
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
            raise ValueError(msg)
        return new_usage + overall_token_usage
    if isinstance(new_usage, dict):
        if not isinstance(overall_token_usage, dict):
            msg = (
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
            raise ValueError(msg)
        return {
            k: _update_token_usage(overall_token_usage.get(k, 0), v)
            for k, v in new_usage.items()
        }
    warnings.warn(f"Unexpected type for token usage: {type(new_usage)}")
    return new_usage


def _handle_openai_bad_request(e: openai.BadRequestError) -> None:
    if (
        "'response_format' of type 'json_schema' is not supported with this model"
    ) in e.message:
        message = (
            "This model does not support OpenAI's structured output feature, which "
            "is the default method for `with_structured_output` as of "
            "langchain-openai==0.3. To use `with_structured_output` with this model, "
            'specify `method="function_calling"`.'
        )
        warnings.warn(message)
        raise e
    if "Invalid schema for response_format" in e.message:
        message = (
            "Invalid schema for OpenAI's structured output feature, which is the "
            "default method for `with_structured_output` as of langchain-openai==0.3. "
            'Specify `method="function_calling"` instead or update your schema. '
            "See supported schemas: "
            "https://platform.openai.com/docs/guides/structured-outputs#supported-schemas"
        )
        warnings.warn(message)
        raise e
    raise


def _model_prefers_responses_api(model_name: str | None) -> bool:
    if not model_name:
        return False
    return "gpt-5.2-pro" in model_name


_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[_BM] | type
_DictOrPydantic: TypeAlias = dict | _BM


class BaseChatOpenAI(BaseChatModel):
    """Base wrapper around OpenAI large language models for chat."""

    client: Any = Field(default=None, exclude=True)

    async_client: Any = Field(default=None, exclude=True)

    root_client: Any = Field(default=None, exclude=True)

    root_async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """Model name to use."""

    temperature: float | None = None
    """What sampling temperature to use."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    openai_api_key: (
        SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]]
    ) = Field(
        alias="api_key", default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
    """API key to use.

    Can be inferred from the `OPENAI_API_KEY` environment variable, or specified as a
    string, or sync or async callable that returns a string.

    ??? example "Specify with environment variable"

        ```bash
        export OPENAI_API_KEY=...
        ```
        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-5-nano")
        ```

    ??? example "Specify with a string"

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-5-nano", api_key="...")
        ```

    ??? example "Specify with a sync callable"
        ```python
        from langchain_openai import ChatOpenAI

        def get_api_key() -> str:
            # Custom logic to retrieve API key
            return "..."

        model = ChatOpenAI(model="gpt-5-nano", api_key=get_api_key)
        ```

    ??? example "Specify with an async callable"
        ```python
        from langchain_openai import ChatOpenAI

        async def get_api_key() -> str:
            # Custom async logic to retrieve API key
            return "..."

        model = ChatOpenAI(model="gpt-5-nano", api_key=get_api_key)
        ```
    """

    openai_api_base: str | None = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service emulator."""  # noqa: E501

    openai_organization: str | None = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""

    # to support explicit proxy for OpenAI
    openai_proxy: str | None = Field(
        default_factory=from_env("OPENAI_PROXY", default=None)
    )

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, `httpx.Timeout` or
    `None`."""

    stream_usage: bool | None = None
    """Whether to include usage metadata in streaming output. If enabled, an additional
    message chunk will be generated during the stream including usage metadata.

    This parameter is enabled unless `openai_api_base` is set or the model is
    initialized with a custom client, as many chat completions APIs do not support
    streaming token usage.

    !!! version-added "Added in `langchain-openai` 0.3.9"

    !!! warning "Behavior changed in `langchain-openai` 0.3.35"

        Enabled for default base URL and client.
    """

    max_retries: int | None = None
    """Maximum number of retries to make when generating."""

    presence_penalty: float | None = None
    """Penalizes repeated tokens."""

    frequency_penalty: float | None = None
    """Penalizes repeated tokens according to frequency."""

    seed: int | None = None
    """Seed for generation"""

    logprobs: bool | None = None
    """Whether to return logprobs."""

    top_logprobs: int | None = None
    """Number of most likely tokens to return at each token position, each with an
    associated log probability. `logprobs` must be set to true if this parameter is
    used."""

    logit_bias: dict[int, int] | None = None
    """Modify the likelihood of specified tokens appearing in the completion."""

    streaming: bool = False
    """Whether to stream the results or not."""

    n: int | None = None
    """Number of chat completions to generate for each prompt."""

    top_p: float | None = None
    """Total probability mass of tokens to consider at each step."""

    max_tokens: int | None = Field(default=None)
    """Maximum number of tokens to generate."""

    reasoning_effort: str | None = None
    """Constrains effort on reasoning for reasoning models. For use with the Chat
    Completions API.

    Reasoning models only.

    Currently supported values are `'minimal'`, `'low'`, `'medium'`, and
    `'high'`. Reducing reasoning effort can result in faster responses and fewer
    tokens used on reasoning in a response.
    """

    reasoning: dict[str, Any] | None = None
    """Reasoning parameters for reasoning models. For use with the Responses API.

    ```python
    reasoning={
        "effort": "medium",  # Can be "low", "medium", or "high"
        "summary": "auto",  # Can be "auto", "concise", or "detailed"
    }
    ```

    !!! version-added "Added in `langchain-openai` 0.3.24"
    """

    verbosity: str | None = None
    """Controls the verbosity level of responses for reasoning models. For use with the
    Responses API.

    Currently supported values are `'low'`, `'medium'`, and `'high'`.

    !!! version-added "Added in `langchain-openai` 0.3.28"
    """

    tiktoken_model_name: str | None = None
    """The model name to pass to tiktoken when using this class.

    Tiktoken is used to count the number of tokens in documents to constrain
    them to be under a certain limit.

    By default, when set to `None`, this will be the same as the embedding model name.
    However, there are some cases where you may want to use this `Embedding` class with
    a model name not supported by tiktoken. This can include when using Azure embeddings
    or when using one of the many model providers that expose an OpenAI-like
    API but with different models. In those cases, in order to avoid erroring
    when tiktoken is called, you can specify a model name to use here.
    """

    default_headers: Mapping[str, str] | None = None

    default_query: Mapping[str, object] | None = None

    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Any | None = Field(default=None, exclude=True)
    """Optional `httpx.Client`.

    Only used for sync invocations. Must specify `http_async_client` as well if you'd
    like a custom client for async invocations.
    """

    http_async_client: Any | None = Field(default=None, exclude=True)
    """Optional `httpx.AsyncClient`.

    Only used for async invocations. Must specify `http_client` as well if you'd like a
    custom client for sync invocations.
    """

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    extra_body: Mapping[str, Any] | None = None
    """Optional additional JSON properties to include in the request parameters when
    making requests to OpenAI compatible APIs, such as vLLM, LM Studio, or other
    providers.

    This is the recommended way to pass custom parameters that are specific to your
    OpenAI-compatible API provider but not part of the standard OpenAI API.

    Examples:
        - [LM Studio](https://lmstudio.ai/) TTL parameter: `extra_body={"ttl": 300}`
        - [vLLM](https://github.com/vllm-project/vllm) custom parameters:
            `extra_body={"use_beam_search": True}`
        - Any other provider-specific parameters

    !!! warning
        Do not use `model_kwargs` for custom parameters that are not part of the
        standard OpenAI API, as this will cause errors when making API calls. Use
        `extra_body` instead.
    """

    include_response_headers: bool = False
    """Whether to include response headers in the output message `response_metadata`."""

    disabled_params: dict[str, Any] | None = Field(default=None)
    """Parameters of the OpenAI client or `chat.completions` endpoint that should be
    disabled for the given model.

    Should be specified as `{"param": None | ['val1', 'val2']}` where the key is the
    parameter and the value is either None, meaning that parameter should never be
    used, or it's a list of disabled values for the parameter.

    For example, older models may not support the `'parallel_tool_calls'` parameter at
    all, in which case `disabled_params={"parallel_tool_calls": None}` can be passed
    in.

    If a parameter is disabled then it will not be used by default in any methods, e.g.
    in `with_structured_output`. However this does not prevent a user from directly
    passed in the parameter during invocation.
    """

    include: list[str] | None = None
    """Additional fields to include in generations from Responses API.

    Supported values:

    - `'file_search_call.results'`
    - `'message.input_image.image_url'`
    - `'computer_call_output.output.image_url'`
    - `'reasoning.encrypted_content'`
    - `'code_interpreter_call.outputs'`

    !!! version-added "Added in `langchain-openai` 0.3.24"
    """

    service_tier: str | None = None
    """Latency tier for request.

    Options are `'auto'`, `'default'`, or `'flex'`.

    Relevant for users of OpenAI's scale tier service.
    """

    store: bool | None = None
    """If `True`, OpenAI may store response data for future use.

    Defaults to `True` for the Responses API and `False` for the Chat Completions API.

    !!! version-added "Added in `langchain-openai` 0.3.24"
    """

    truncation: str | None = None
    """Truncation strategy (Responses API).

    Can be `'auto'` or `'disabled'` (default).

    If `'auto'`, model may drop input items from the middle of the message sequence to
    fit the context window.

    !!! version-added "Added in `langchain-openai` 0.3.24"
    """

    use_previous_response_id: bool = False
    """If `True`, always pass `previous_response_id` using the ID of the most recent
    response. Responses API only.

    Input messages up to the most recent response will be dropped from request
    payloads.

    For example, the following two are equivalent:

    ```python
    model = ChatOpenAI(
        model="...",
        use_previous_response_id=True,
    )
    model.invoke(
        [
            HumanMessage("Hello"),
            AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
            HumanMessage("How are you?"),
        ]
    )
    ```

    ```python
    model = ChatOpenAI(model="...", use_responses_api=True)
    model.invoke([HumanMessage("How are you?")], previous_response_id="resp_123")
    ```

    !!! version-added "Added in `langchain-openai` 0.3.26"
    """

    use_responses_api: bool | None = None
    """Whether to use the Responses API instead of the Chat API.

    If not specified then will be inferred based on invocation params.

    !!! version-added "Added in `langchain-openai` 0.3.9"
    """

    output_version: str | None = Field(
        default_factory=from_env("LC_OUTPUT_VERSION", default=None)
    )
    """Version of `AIMessage` output format to use.

    This field is used to roll-out new output formats for chat model `AIMessage`
    responses in a backwards-compatible way.

    Supported values:

    - `'v0'`: `AIMessage` format as of `langchain-openai 0.3.x`.
    - `'responses/v1'`: Formats Responses API output items into AIMessage content blocks
        (Responses API only)
    - `'v1'`: v1 of LangChain cross-provider standard.

    !!! warning "Behavior changed in `langchain-openai` 1.0.0"

        Default updated to `"responses/v1"`.
    """

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="before")
    @classmethod
    def validate_temperature(cls, values: dict[str, Any]) -> Any:
        """Validate temperature parameter for different models.

        - gpt-5 models (excluding gpt-5-chat) only allow `temperature=1` or unset
            (Defaults to 1)
        """
        model = values.get("model_name") or values.get("model") or ""
        model_lower = model.lower()

        # For o1 models, set temperature=1 if not provided
        if model_lower.startswith("o1") and "temperature" not in values:
            values["temperature"] = 1

        # For gpt-5 models, handle temperature restrictions. Temperature is supported
        # by gpt-5-chat and gpt-5 models with reasoning_effort='none' or
        # reasoning={'effort': 'none'}.
        if (
            model_lower.startswith("gpt-5")
            and ("chat" not in model_lower)
            and values.get("reasoning_effort") != "none"
            and (values.get("reasoning") or {}).get("effort") != "none"
        ):
            temperature = values.get("temperature")
            if temperature is not None and temperature != 1:
                # For gpt-5 (non-chat), only temperature=1 is supported
                # So we remove any non-defaults
                values.pop("temperature", None)

        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        # Check OPENAI_ORGANIZATION for backwards compatibility.
        self.openai_organization = (
            self.openai_organization
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        self.openai_api_base = self.openai_api_base or os.getenv("OPENAI_API_BASE")

        # Enable stream_usage by default if using default base URL and client
        if (
            all(
                getattr(self, key, None) is None
                for key in (
                    "stream_usage",
                    "openai_proxy",
                    "openai_api_base",
                    "base_url",
                    "client",
                    "root_client",
                    "async_client",
                    "root_async_client",
                    "http_client",
                    "http_async_client",
                )
            )
            and "OPENAI_BASE_URL" not in os.environ
        ):
            self.stream_usage = True

        # Resolve API key from SecretStr or Callable
        sync_api_key_value: str | Callable[[], str] | None = None
        async_api_key_value: str | Callable[[], Awaitable[str]] | None = None

        if self.openai_api_key is not None:
            # Because OpenAI and AsyncOpenAI clients support either sync or async
            # callables for the API key, we need to resolve separate values here.
            sync_api_key_value, async_api_key_value = _resolve_sync_and_async_api_keys(
                self.openai_api_key
            )

        client_params: dict = {
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if self.openai_proxy and (self.http_client or self.http_async_client):
            openai_proxy = self.openai_proxy
            http_client = self.http_client
            http_async_client = self.http_async_client
            msg = (
                "Cannot specify 'openai_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{openai_proxy=}\n{http_client=}\n{http_async_client=}"
            )
            raise ValueError(msg)
        if not self.client:
            if sync_api_key_value is None:
                # No valid sync API key, leave client as None and raise informative
                # error on invocation.
                self.client = None
                self.root_client = None
            else:
                if self.openai_proxy and not self.http_client:
                    try:
                        import httpx
                    except ImportError as e:
                        msg = (
                            "Could not import httpx python package. "
                            "Please install it with `pip install httpx`."
                        )
                        raise ImportError(msg) from e
                    self.http_client = httpx.Client(
                        proxy=self.openai_proxy, verify=global_ssl_context
                    )
                sync_specific = {
                    "http_client": self.http_client
                    or _get_default_httpx_client(
                        self.openai_api_base, self.request_timeout
                    ),
                    "api_key": sync_api_key_value,
                }
                self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
                self.client = self.root_client.chat.completions
        if not self.async_client:
            if self.openai_proxy and not self.http_async_client:
                try:
                    import httpx
                except ImportError as e:
                    msg = (
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    )
                    raise ImportError(msg) from e
                self.http_async_client = httpx.AsyncClient(
                    proxy=self.openai_proxy, verify=global_ssl_context
                )
            async_specific = {
                "http_client": self.http_async_client
                or _get_default_async_httpx_client(
                    self.openai_api_base, self.request_timeout
                ),
                "api_key": async_api_key_value,
            }
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model_name)
        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        exclude_if_none = {
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "logit_bias": self.logit_bias,
            "stop": self.stop or None,  # Also exclude empty list for this
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
            "n": self.n,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "reasoning": self.reasoning,
            "verbosity": self.verbosity,
            "include": self.include,
            "service_tier": self.service_tier,
            "truncation": self.truncation,
            "store": self.store,
        }

        return {
            "model": self.model_name,
            "stream": self.streaming,
            **{k: v for k, v in exclude_if_none.items() if v is not None},
            **self.model_kwargs,
        }

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output.get("token_usage")
            if token_usage is not None:
                for k, v in token_usage.items():
                    if v is None:
                        continue
                    if k in overall_token_usage:
                        overall_token_usage[k] = _update_token_usage(
                            overall_token_usage[k], v
                        )
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        if chunk.get("type") == "content.delta":  # From beta.chat.completions.stream
            return None
        token_usage = chunk.get("usage")
        choices = (
            chunk.get("choices", [])
            # From beta.chat.completions.stream
            or chunk.get("chunk", {}).get("choices", [])
        )

        usage_metadata: UsageMetadata | None = (
            _create_usage_metadata(token_usage, chunk.get("service_tier"))
            if token_usage
            else None
        )
        if len(choices) == 0:
            # logprobs is implicitly None
            generation_chunk = ChatGenerationChunk(
                message=default_chunk_class(content="", usage_metadata=usage_metadata),
                generation_info=base_generation_info,
            )
            if self.output_version == "v1":
                generation_chunk.message.content = []
                generation_chunk.message.response_metadata["output_version"] = "v1"

            return generation_chunk

        choice = choices[0]
        if choice["delta"] is None:
            return None

        message_chunk = _convert_delta_to_message_chunk(
            choice["delta"], default_chunk_class
        )
        generation_info = {**base_generation_info} if base_generation_info else {}

        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason
            if model_name := chunk.get("model"):
                generation_info["model_name"] = model_name
            if system_fingerprint := chunk.get("system_fingerprint"):
                generation_info["system_fingerprint"] = system_fingerprint
            if service_tier := chunk.get("service_tier"):
                generation_info["service_tier"] = service_tier

        logprobs = choice.get("logprobs")
        if logprobs:
            generation_info["logprobs"] = logprobs

        if usage_metadata and isinstance(message_chunk, AIMessageChunk):
            message_chunk.usage_metadata = usage_metadata

        message_chunk.response_metadata["model_provider"] = "openai"
        return ChatGenerationChunk(
            message=message_chunk, generation_info=generation_info or None
        )

    def _ensure_sync_client_available(self) -> None:
        """Check that sync client is available, raise error if not."""
        if self.client is None:
            msg = (
                "Sync client is not available. This happens when an async callable "
                "was provided for the API key. Use async methods (ainvoke, astream) "
                "instead, or provide a string or sync callable for the API key."
            )
            raise ValueError(msg)

    def _stream_responses(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._ensure_sync_client_available()
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if self.include_response_headers:
            raw_context_manager = self.root_client.with_raw_response.responses.create(
                **payload
            )
            context_manager = raw_context_manager.parse()
            headers = {"headers": dict(raw_context_manager.headers)}
        else:
            context_manager = self.root_client.responses.create(**payload)
            headers = {}
        original_schema_obj = kwargs.get("response_format")

        with context_manager as response:
            is_first_chunk = True
            current_index = -1
            current_output_index = -1
            current_sub_index = -1
            has_reasoning = False
            for chunk in response:
                metadata = headers if is_first_chunk else {}
                (
                    current_index,
                    current_output_index,
                    current_sub_index,
                    generation_chunk,
                ) = _convert_responses_chunk_to_generation_chunk(
                    chunk,
                    current_index,
                    current_output_index,
                    current_sub_index,
                    schema=original_schema_obj,
                    metadata=metadata,
                    has_reasoning=has_reasoning,
                    output_version=self.output_version,
                )
                if generation_chunk:
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text, chunk=generation_chunk
                        )
                    is_first_chunk = False
                    if "reasoning" in generation_chunk.message.additional_kwargs:
                        has_reasoning = True
                    yield generation_chunk

    async def _astream_responses(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if self.include_response_headers:
            raw_context_manager = (
                await self.root_async_client.with_raw_response.responses.create(
                    **payload
                )
            )
            context_manager = raw_context_manager.parse()
            headers = {"headers": dict(raw_context_manager.headers)}
        else:
            context_manager = await self.root_async_client.responses.create(**payload)
            headers = {}
        original_schema_obj = kwargs.get("response_format")

        async with context_manager as response:
            is_first_chunk = True
            current_index = -1
            current_output_index = -1
            current_sub_index = -1
            has_reasoning = False
            async for chunk in response:
                metadata = headers if is_first_chunk else {}
                (
                    current_index,
                    current_output_index,
                    current_sub_index,
                    generation_chunk,
                ) = _convert_responses_chunk_to_generation_chunk(
                    chunk,
                    current_index,
                    current_output_index,
                    current_sub_index,
                    schema=original_schema_obj,
                    metadata=metadata,
                    has_reasoning=has_reasoning,
                    output_version=self.output_version,
                )
                if generation_chunk:
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            generation_chunk.text, chunk=generation_chunk
                        )
                    is_first_chunk = False
                    if "reasoning" in generation_chunk.message.additional_kwargs:
                        has_reasoning = True
                    yield generation_chunk

    def _should_stream_usage(
        self, stream_usage: bool | None = None, **kwargs: Any
    ) -> bool:
        """Determine whether to include usage metadata in streaming output.

        For backwards compatibility, we check for `stream_options` passed
        explicitly to kwargs or in the `model_kwargs` and override `self.stream_usage`.
        """
        stream_usage_sources = [  # order of precedence
            stream_usage,
            kwargs.get("stream_options", {}).get("include_usage"),
            self.model_kwargs.get("stream_options", {}).get("include_usage"),
            self.stream_usage,
        ]
        for source in stream_usage_sources:
            if isinstance(source, bool):
                return source
        return self.stream_usage or False

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._ensure_sync_client_available()
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}

        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            response_stream = self.root_client.beta.chat.completions.stream(**payload)
            context_manager = response_stream
        else:
            if self.include_response_headers:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
            else:
                response = self.client.create(**payload)
            context_manager = response
        try:
            with context_manager as response:
                is_first_chunk = True
                for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(
                final_completion
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_sync_client_available()
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        raw_response = None
        try:
            if "response_format" in payload:
                payload.pop("stream")
                try:
                    raw_response = (
                        self.root_client.chat.completions.with_raw_response.parse(
                            **payload
                        )
                    )
                    response = raw_response.parse()
                except openai.BadRequestError as e:
                    _handle_openai_bad_request(e)
            elif self._use_responses_api(payload):
                original_schema_obj = kwargs.get("response_format")
                if original_schema_obj and _is_pydantic_class(original_schema_obj):
                    raw_response = self.root_client.responses.with_raw_response.parse(
                        **payload
                    )
                else:
                    raw_response = self.root_client.responses.with_raw_response.create(
                        **payload
                    )
                response = raw_response.parse()
                if self.include_response_headers:
                    generation_info = {"headers": dict(raw_response.headers)}
                return _construct_lc_result_from_responses_api(
                    response,
                    schema=original_schema_obj,
                    metadata=generation_info,
                    output_version=self.output_version,
                )
            else:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
        except Exception as e:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                e.response = raw_response.http_response  # type: ignore[attr-defined]
            raise e
        if (
            self.include_response_headers
            and raw_response is not None
            and hasattr(raw_response, "headers")
        ):
            generation_info = {"headers": dict(raw_response.headers)}
        return self._create_chat_result(response, generation_info)

    def _use_responses_api(self, payload: dict) -> bool:
        if isinstance(self.use_responses_api, bool):
            return self.use_responses_api
        if (
            self.output_version == "responses/v1"
            or self.include is not None
            or self.reasoning is not None
            or self.truncation is not None
            or self.use_previous_response_id
            or _model_prefers_responses_api(self.model_name)
        ):
            return True
        return _use_responses_api(payload)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params, **kwargs}

        if self._use_responses_api(payload):
            if self.use_previous_response_id:
                last_messages, previous_response_id = _get_last_messages(messages)
                payload_to_use = last_messages if previous_response_id else messages
                if previous_response_id:
                    payload["previous_response_id"] = previous_response_id
                payload = _construct_responses_api_payload(payload_to_use, payload)
            else:
                payload = _construct_responses_api_payload(messages, payload)
        else:
            payload["messages"] = [
                _convert_message_to_dict(_convert_from_v1_to_chat_completions(m))
                if isinstance(m, AIMessage)
                else _convert_message_to_dict(m)
                for m in messages
            ]
        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        generations = []

        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        # Sometimes the AI Model calling will get error, we should raise it (this is
        # typically followed by a null value for `choices`, which we raise for
        # separately below).
        if response_dict.get("error"):
            raise ValueError(response_dict.get("error"))

        # Raise informative error messages for non-OpenAI chat completions APIs
        # that return malformed responses.
        try:
            choices = response_dict["choices"]
        except KeyError as e:
            msg = f"Response missing `choices` key: {response_dict.keys()}"
            raise KeyError(msg) from e

        if choices is None:
            msg = "Received response with null value for `choices`."
            raise TypeError(msg)

        token_usage = response_dict.get("usage")
        service_tier = response_dict.get("service_tier")

        for res in choices:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(
                    token_usage, service_tier
                )
            generation_info = generation_info or {}
            generation_info["finish_reason"] = (
                res.get("finish_reason")
                if res.get("finish_reason") is not None
                else generation_info.get("finish_reason")
            )
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_provider": "openai",
            "model_name": response_dict.get("model", self.model_name),
            "system_fingerprint": response_dict.get("system_fingerprint", ""),
        }
        if "id" in response_dict:
            llm_output["id"] = response_dict["id"]
        if service_tier:
            llm_output["service_tier"] = service_tier

        if isinstance(response, openai.BaseModel) and getattr(
            response, "choices", None
        ):
            message = response.choices[0].message  # type: ignore[attr-defined]
            if hasattr(message, "parsed"):
                generations[0].message.additional_kwargs["parsed"] = message.parsed
            if hasattr(message, "refusal"):
                generations[0].message.additional_kwargs["refusal"] = message.refusal

        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}

        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            response_stream = self.root_async_client.beta.chat.completions.stream(
                **payload
            )
            context_manager = response_stream
        else:
            if self.include_response_headers:
                raw_response = await self.async_client.with_raw_response.create(
                    **payload
                )
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
            else:
                response = await self.async_client.create(**payload)
            context_manager = response
        try:
            async with context_manager as response:
                is_first_chunk = True
                async for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = await response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(
                final_completion
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        raw_response = None
        try:
            if "response_format" in payload:
                payload.pop("stream")
                try:
                    raw_response = await self.root_async_client.chat.completions.with_raw_response.parse(  # noqa: E501
                        **payload
                    )
                    response = raw_response.parse()
                except openai.BadRequestError as e:
                    _handle_openai_bad_request(e)
            elif self._use_responses_api(payload):
                original_schema_obj = kwargs.get("response_format")
                if original_schema_obj and _is_pydantic_class(original_schema_obj):
                    raw_response = (
                        await self.root_async_client.responses.with_raw_response.parse(
                            **payload
                        )
                    )
                else:
                    raw_response = (
                        await self.root_async_client.responses.with_raw_response.create(
                            **payload
                        )
                    )
                response = raw_response.parse()
                if self.include_response_headers:
                    generation_info = {"headers": dict(raw_response.headers)}
                return _construct_lc_result_from_responses_api(
                    response,
                    schema=original_schema_obj,
                    metadata=generation_info,
                    output_version=self.output_version,
                )
            else:
                raw_response = await self.async_client.with_raw_response.create(
                    **payload
                )
                response = raw_response.parse()
        except Exception as e:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                e.response = raw_response.http_response  # type: ignore[attr-defined]
            raise e
        if (
            self.include_response_headers
            and raw_response is not None
            and hasattr(raw_response, "headers")
        ):
            generation_info = {"headers": dict(raw_response.headers)}
        return await run_in_executor(
            None, self._create_chat_result, response, generation_info
        )

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}

    def _get_invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params = {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }
        # Redact headers from built-in remote MCP tool invocations
        if (tools := params.get("tools")) and isinstance(tools, list):
            params["tools"] = [
                ({**tool, "headers": "**REDACTED**"} if "headers" in tool else tool)
                if isinstance(tool, dict) and tool.get("type") == "mcp"
                else tool
                for tool in tools
            ]

        return params

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="openai",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens) or params.get(
            "max_completion_tokens", self.max_tokens
        ):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model.

        Will always return `'openai-chat'` regardless of the specific model name.
        """
        return "openai-chat"

    def _get_encoding_model(self) -> tuple[str, tiktoken.Encoding]:
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            model_lower = model.lower()
            encoder = "cl100k_base"
            if model_lower.startswith(("gpt-4o", "gpt-4.1", "gpt-5")):
                encoder = "o200k_base"
            encoding = tiktoken.get_encoding(encoder)
        return model, encoding

    def get_token_ids(self, text: str) -> list[int]:
        """Get the tokens present in the text with tiktoken package."""
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    def get_num_tokens_from_messages(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool] | None = None,
    ) -> int:
        """Calculate num tokens for `gpt-3.5-turbo` and `gpt-4` with `tiktoken` package.

        !!! warning
            You must have the `pillow` installed if you want to count image tokens if
            you are specifying the image as a base64 string, and you must have both
            `pillow` and `httpx` installed if you are specifying the image as a URL. If
            these aren't installed image inputs will be ignored in token counting.

        [OpenAI reference](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb).

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of `dict`, `BaseModel`, function, or `BaseTool`
                to be converted to tool schemas.
        """
        # TODO: Count bound tools as part of input.
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools."
            )
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("gpt-3.5-turbo-0301"):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model.startswith(("gpt-3.5-turbo", "gpt-4", "gpt-5")):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            msg = (
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}. See "
                "https://platform.openai.com/docs/guides/text-generation/managing-tokens"
                " for information on how messages are converted to tokens."
            )
            raise NotImplementedError(msg)
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # This is an inferred approximation. OpenAI does not document how to
                # count tool message tokens.
                if key == "tool_call_id":
                    num_tokens += 3
                    continue
                if isinstance(value, list):
                    # content or tool calls
                    for val in value:
                        if isinstance(val, str) or val["type"] == "text":
                            text = val["text"] if isinstance(val, dict) else val
                            num_tokens += len(encoding.encode(text))
                        elif val["type"] == "image_url":
                            if val["image_url"].get("detail") == "low":
                                num_tokens += 85
                            else:
                                image_size = _url_to_size(val["image_url"]["url"])
                                if not image_size:
                                    continue
                                num_tokens += _count_image_tokens(*image_size)
                        # Tool/function call token counting is not documented by OpenAI.
                        # This is an approximation.
                        elif val["type"] == "function":
                            num_tokens += len(
                                encoding.encode(val["function"]["arguments"])
                            )
                            num_tokens += len(encoding.encode(val["function"]["name"]))
                        elif val["type"] == "file":
                            warnings.warn(
                                "Token counts for file inputs are not supported. "
                                "Ignoring file inputs."
                            )
                        else:
                            msg = f"Unrecognized content block type\n\n{val}"
                            raise ValueError(msg)
                elif not value:
                    continue
                else:
                    # Cast str(value) in case the message value is not a string
                    # This occurs with function messages
                    num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: _DictOrPydanticClass | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.

                Supports any tool definition handled by [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool].
            tool_choice: Which tool to require the model to call. Options are:

                - `str` of the form `'<<tool_name>>'`: calls `<<tool_name>>` tool.
                - `'auto'`: automatically selects a tool (including no tool).
                - `'none'`: does not call a tool.
                - `'any'` or `'required'` or `True`: force at least one tool to be called.
                - `dict` of the form `{"type": "function", "function": {"name": <<tool_name>>}}`: calls `<<tool_name>>` tool.
                - `False` or `None`: no effect, default OpenAI behavior.
            strict: If `True`, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition. The input schema will also be validated according to the
                [supported schemas](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas?api-mode=responses#supported-schemas).
                If `False`, input schema will not be validated and model output will not
                be validated. If `None`, `strict` argument will not be passed to the model.
            parallel_tool_calls: Set to `False` to disable parallel tool use.
                Defaults to `None` (no specification, which allows parallel tool use).
            response_format: Optional schema to format model response. If provided
                and the model does not call a tool, the model will generate a
                [structured response](https://platform.openai.com/docs/guides/structured-outputs).
            kwargs: Any additional parameters are passed directly to `bind`.
        """  # noqa: E501
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])
            else:
                pass
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice in WellKnownTools:
                    tool_choice = {"type": tool_choice}
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                elif tool_choice == "any":
                    tool_choice = "required"
                else:
                    pass
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                pass
            else:
                msg = (
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
            kwargs["tool_choice"] = tool_choice

        if response_format:
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and "schema" in response_format.get("json_schema", {})
            ):
                # compat with langchain.agents.create_agent response_format, which is
                # an approximation of OpenAI format
                strict = response_format["json_schema"].get("strict", None)
                response_format = cast(dict, response_format["json_schema"]["schema"])
            kwargs["response_format"] = _convert_to_openai_response_format(
                response_format, strict=strict
            )
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        tools: list | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
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
                    Uses OpenAI's [tool-calling API](https://platform.openai.com/docs/guides/function-calling)
                    (formerly called function calling)
                - `'json_schema'`:
                    Uses OpenAI's [Structured Output API](https://platform.openai.com/docs/guides/structured-outputs)
                - `'json_mode'`:
                    Uses OpenAI's [JSON mode](https://platform.openai.com/docs/guides/structured-outputs/json-mode).
                    Note that if using JSON mode then you must include instructions for
                    formatting the output into the desired schema into the model call

            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:

                - `True`:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the
                    [supported schemas](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas?api-mode=responses#supported-schemas).
                - `False`:
                    Input schema will not be validated and model output will not be
                    validated.
                - `None`:
                    `strict` argument will not be passed to the model.

            tools:
                A list of tool-like objects to bind to the chat model. Requires that:

                - `method` is `'json_schema'` (default).
                - `strict=True`
                - `include_raw=True`

                If a model elects to call a tool, the resulting `AIMessage` in `'raw'`
                will include tool calls.

                ??? example

                    ```python
                    from langchain.chat_models import init_chat_model
                    from pydantic import BaseModel


                    class ResponseSchema(BaseModel):
                        response: str


                    def get_weather(location: str) -> str:
                        \"\"\"Get weather at a location.\"\"\"
                        pass

                    model = init_chat_model("openai:gpt-4o-mini")

                    structured_model = model.with_structured_output(
                        ResponseSchema,
                        tools=[get_weather],
                        strict=True,
                        include_raw=True,
                    )

                    structured_model.invoke("What's the weather in Boston?")
                    ```

                    ```python
                    {
                        "raw": AIMessage(content="", tool_calls=[...], ...),
                        "parsing_error": None,
                        "parsed": None,
                    }
                    ```

            kwargs: Additional keyword args are passed through to the model.

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

        !!! warning "Behavior changed in `langchain-openai` 0.3.12"

            Support for `tools` added.

        !!! warning "Behavior changed in `langchain-openai` 0.3.21"

            Pass `kwargs` through to the model.
        """
        if strict is not None and method == "json_mode":
            msg = "Argument `strict` is not supported with `method`='json_mode'"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)

        if method == "json_schema":
            # Check for Pydantic BaseModel V1
            if (
                is_pydantic_schema and issubclass(schema, BaseModelV1)  # type: ignore[arg-type]
            ):
                warnings.warn(
                    "Received a Pydantic BaseModel V1 schema. This is not supported by "
                    'method="json_schema". Please use method="function_calling" '
                    "or specify schema via JSON Schema or Pydantic V2 BaseModel. "
                    'Overriding to method="function_calling".'
                )
                method = "function_calling"
            # Check for incompatible model
            if self.model_name and (
                self.model_name.startswith("gpt-3")
                or self.model_name.startswith("gpt-4-")
                or self.model_name == "gpt-4"
            ):
                warnings.warn(
                    f"Cannot use method='json_schema' with model {self.model_name} "
                    f"since it doesn't support OpenAI's Structured Output API. You can "
                    f"see supported models here: "
                    f"https://platform.openai.com/docs/guides/structured-outputs#supported-models. "  # noqa: E501
                    "To fix this warning, set `method='function_calling'. "
                    "Overriding to method='function_calling'."
                )
                method = "function_calling"

        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            bind_kwargs = self._filter_disabled_params(
                **{
                    "tool_choice": tool_name,
                    "parallel_tool_calls": False,
                    "strict": strict,
                    "ls_structured_output_format": {
                        "kwargs": {"method": method, "strict": strict},
                        "schema": schema,
                    },
                    **kwargs,
                }
            )

            llm = self.bind_tools([schema], **bind_kwargs)
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                **{
                    "response_format": {"type": "json_object"},
                    "ls_structured_output_format": {
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                    **kwargs,
                }
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            response_format = _convert_to_openai_response_format(schema, strict=strict)
            bind_kwargs = {
                **dict(
                    response_format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method, "strict": strict},
                        "schema": convert_to_openai_tool(schema),
                    },
                    **kwargs,
                )
            }
            if tools:
                bind_kwargs["tools"] = [
                    convert_to_openai_tool(t, strict=strict) for t in tools
                ]
            llm = self.bind(**bind_kwargs)
            if is_pydantic_schema:
                output_parser = RunnableLambda(
                    partial(_oai_structured_outputs_parser, schema=cast(type, schema))
                ).with_types(output_type=cast(type, schema))
            else:
                output_parser = JsonOutputParser()
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

    def _filter_disabled_params(self, **kwargs: Any) -> dict[str, Any]:
        if not self.disabled_params:
            return kwargs
        filtered = {}
        for k, v in kwargs.items():
            # Skip param
            if k in self.disabled_params and (
                self.disabled_params[k] is None or v in self.disabled_params[k]
            ):
                continue
            # Keep param
            filtered[k] = v
        return filtered

    def _get_generation_chunk_from_completion(
        self, completion: openai.BaseModel
    ) -> ChatGenerationChunk:
        """Get chunk from completion (e.g., from final completion of a stream)."""
        chat_result = self._create_chat_result(completion)
        chat_message = chat_result.generations[0].message
        if isinstance(chat_message, AIMessage):
            usage_metadata = chat_message.usage_metadata
            # Skip tool_calls, already sent as chunks
            if "tool_calls" in chat_message.additional_kwargs:
                chat_message.additional_kwargs.pop("tool_calls")
        else:
            usage_metadata = None
        message = AIMessageChunk(
            content="",
            additional_kwargs=chat_message.additional_kwargs,
            usage_metadata=usage_metadata,
        )
        return ChatGenerationChunk(
            message=message, generation_info=chat_result.llm_output
        )


class ChatOpenAI(BaseChatOpenAI):  # type: ignore[override]
    r"""Interface to OpenAI chat model APIs.

    ???+ info "Setup"

        Install `langchain-openai` and set environment variable `OPENAI_API_KEY`.

        ```bash
        pip install -U langchain-openai

        # or using uv
        uv add langchain-openai
        ```

        ```bash
        export OPENAI_API_KEY="your-api-key"
        ```

    ??? info "Key init args — completion params"

        | Param               | Type          | Description                                                                                                 |
        | ------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
        | `model`             | `str`         | Name of OpenAI model to use.                                                                                |
        | `temperature`       | `float`       | Sampling temperature.                                                                                       |
        | `max_tokens`        | `int | None`  | Max number of tokens to generate.                                                                           |
        | `logprobs`          | `bool | None` | Whether to return logprobs.                                                                                 |
        | `stream_options`    | `dict`        | Configure streaming outputs, like whether to return token usage when streaming (`{"include_usage": True}`). |
        | `use_responses_api` | `bool | None` | Whether to use the responses API.                                                                           |

        See full list of supported init args and their descriptions below.

    ??? info "Key init args — client params"

        | Param          | Type                                       | Description                                                                         |
        | -------------- | ------------------------------------------ | ----------------------------------------------------------------------------------- |
        | `timeout`      | `float | Tuple[float, float] | Any | None` | Timeout for requests.                                                               |
        | `max_retries`  | `int | None`                               | Max number of retries.                                                              |
        | `api_key`      | `str | None`                               | OpenAI API key. If not passed in will be read from env var `OPENAI_API_KEY`.        |
        | `base_url`     | `str | None`                               | Base URL for API requests. Only specify if using a proxy or service emulator.       |
        | `organization` | `str | None`                               | OpenAI organization ID. If not passed in will be read from env var `OPENAI_ORG_ID`. |

        See full list of supported init args and their descriptions below.

    ??? info "Instantiate"

        Create a model instance with desired params. For example:

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="...",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # base_url="...",
            # organization="...",
            # other params...
        )
        ```

        See all available params below.

        !!! tip "Preserved params"
            Any param which is not explicitly supported will be passed directly to
            [`openai.OpenAI.chat.completions.create(...)`](https://platform.openai.com/docs/api-reference/chat/create)
            every time to the model is invoked. For example:

            ```python
            from langchain_openai import ChatOpenAI
            import openai

            ChatOpenAI(..., frequency_penalty=0.2).invoke(...)

            # Results in underlying API call of:

            openai.OpenAI(..).chat.completions.create(..., frequency_penalty=0.2)

            # Which is also equivalent to:

            ChatOpenAI(...).invoke(..., frequency_penalty=0.2)
            ```

    ??? info "Invoke"

        Generate a response from the model:

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

        Results in an `AIMessage` response:

        ```python
        AIMessage(
            content="J'adore la programmation.",
            response_metadata={
                "token_usage": {
                    "completion_tokens": 5,
                    "prompt_tokens": 31,
                    "total_tokens": 36,
                },
                "model_name": "gpt-4o",
                "system_fingerprint": "fp_43dfabdef1",
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-012cffe2-5d3d-424d-83b5-51c6d4a593d1-0",
            usage_metadata={"input_tokens": 31, "output_tokens": 5, "total_tokens": 36},
        )
        ```

    ??? info "Stream"

        Stream a response from the model:

        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

        Results in a sequence of `AIMessageChunk` objects with partial content:

        ```python
        AIMessageChunk(content="", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(content="J", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(content="'adore", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(content=" la", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(
            content=" programmation", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0"
        )
        AIMessageChunk(content=".", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "stop"},
            id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0",
        )
        ```

        To collect the full message, you can concatenate the chunks:

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        ```

        ```python
        full = AIMessageChunk(
            content="J'adore la programmation.",
            response_metadata={"finish_reason": "stop"},
            id="run-bf917526-7f58-4683-84f7-36a6b671d140",
        )
        ```

    ??? info "Async"

        Asynchronous equivalents of `invoke`, `stream`, and `batch` are also available:

        ```python
        # Invoke
        await model.ainvoke(messages)

        # Stream
        async for chunk in (await model.astream(messages))

        # Batch
        await model.abatch([messages])
        ```

        Results in an `AIMessage` response:

        ```python
        AIMessage(
            content="J'adore la programmation.",
            response_metadata={
                "token_usage": {
                    "completion_tokens": 5,
                    "prompt_tokens": 31,
                    "total_tokens": 36,
                },
                "model_name": "gpt-4o",
                "system_fingerprint": "fp_43dfabdef1",
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-012cffe2-5d3d-424d-83b5-51c6d4a593d1-0",
            usage_metadata={
                "input_tokens": 31,
                "output_tokens": 5,
                "total_tokens": 36,
            },
        )
        ```

        For batched calls, results in a `list[AIMessage]`.

    ??? info "Tool calling"

        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        model_with_tools = model.bind_tools(
            [GetWeather, GetPopulation]
            # strict = True  # Enforce tool args schema is respected
        )
        ai_msg = model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetWeather",
                "args": {"location": "Los Angeles, CA"},
                "id": "call_6XswGD5Pqk8Tt5atYr7tfenU",
            },
            {
                "name": "GetWeather",
                "args": {"location": "New York, NY"},
                "id": "call_ZVL15vA8Y7kXqOy3dtmQgeCi",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "call_49CFW8zqC9W7mh7hbMLSIrXw",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "New York, NY"},
                "id": "call_6ghfKxV264jEfe1mRIkS3PE7",
            },
        ]
        ```

        !!! note "Parallel tool calls"
            [`openai >= 1.32`](https://pypi.org/project/openai/) supports a
            `parallel_tool_calls` parameter that defaults to `True`. This parameter can
            be set to `False` to disable parallel tool calls:

            ```python
            ai_msg = model_with_tools.invoke(
                "What is the weather in LA and NY?", parallel_tool_calls=False
            )
            ai_msg.tool_calls
            ```

            ```python
            [
                {
                    "name": "GetWeather",
                    "args": {"location": "Los Angeles, CA"},
                    "id": "call_4OoY0ZR99iEvC7fevsH8Uhtz",
                }
            ]
            ```

        Like other runtime parameters, `parallel_tool_calls` can be bound to a model
        using `model.bind(parallel_tool_calls=False)` or during instantiation by
        setting `model_kwargs`.

        See `bind_tools` for more.

    ??? info "Built-in (server-side) tools"

        You can access [built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses)
        supported by the OpenAI Responses API. See [LangChain docs](https://docs.langchain.com/oss/python/integrations/chat/openai#responses-api)
        for more detail.

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="...", output_version="responses/v1")

        tool = {"type": "web_search"}
        model_with_tools = model.bind_tools([tool])

        response = model_with_tools.invoke("What was a positive news story from today?")
        response.content
        ```

        ```python
        [
            {
                "type": "text",
                "text": "Today, a heartwarming story emerged from ...",
                "annotations": [
                    {
                        "end_index": 778,
                        "start_index": 682,
                        "title": "Title of story",
                        "type": "url_citation",
                        "url": "<url of story>",
                    }
                ],
            }
        ]
        ```

        !!! version-added "Added in `langchain-openai` 0.3.9"

        !!! version-added "Added in `langchain-openai` 0.3.26: Updated `AIMessage` format"
            [`langchain-openai >= 0.3.26`](https://pypi.org/project/langchain-openai/#history)
            allows users to opt-in to an updated `AIMessage` format when using the
            Responses API. Setting `ChatOpenAI(..., output_version="responses/v1")` will
            format output from reasoning summaries, built-in tool invocations, and other
            response items into the message's `content` field, rather than
            `additional_kwargs`. We recommend this format for new applications.

    ??? info "Managing conversation state"

        OpenAI's Responses API supports management of [conversation state](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses).
        Passing in response IDs from previous messages will continue a conversational
        thread.

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="...",
            use_responses_api=True,
            output_version="responses/v1",
        )
        response = model.invoke("Hi, I'm Bob.")
        response.text
        ```

        ```txt
        "Hi Bob! How can I assist you today?"
        ```

        ```python
        second_response = model.invoke(
            "What is my name?",
            previous_response_id=response.response_metadata["id"],
        )
        second_response.text
        ```

        ```txt
        "Your name is Bob. How can I help you today, Bob?"
        ```

        !!! version-added "Added in `langchain-openai` 0.3.9"

        !!! version-added "Added in `langchain-openai` 0.3.26"
            You can also initialize `ChatOpenAI` with `use_previous_response_id`.
            Input messages up to the most recent response will then be dropped from request
            payloads, and `previous_response_id` will be set using the ID of the most
            recent response.

            ```python
            model = ChatOpenAI(model="...", use_previous_response_id=True)
            ```

    ??? info "Reasoning output"

        OpenAI's Responses API supports [reasoning models](https://platform.openai.com/docs/guides/reasoning?api-mode=responses)
        that expose a summary of internal reasoning processes.

        ```python
        from langchain_openai import ChatOpenAI

        reasoning = {
            "effort": "medium",  # 'low', 'medium', or 'high'
            "summary": "auto",  # 'detailed', 'auto', or None
        }

        model = ChatOpenAI(
            model="...", reasoning=reasoning, output_version="responses/v1"
        )
        response = model.invoke("What is 3^3?")

        # Response text
        print(f"Output: {response.text}")

        # Reasoning summaries
        for block in response.content:
            if block["type"] == "reasoning":
                for summary in block["summary"]:
                    print(summary["text"])
        ```

        ```txt
        Output: 3³ = 27
        Reasoning: The user wants to know...
        ```

        !!! version-added "Added in `langchain-openai` 0.3.26: Updated `AIMessage` format"
            [`langchain-openai >= 0.3.26`](https://pypi.org/project/langchain-openai/#history)
            allows users to opt-in to an updated `AIMessage` format when using the
            Responses API. Setting `ChatOpenAI(..., output_version="responses/v1")` will
            format output from reasoning summaries, built-in tool invocations, and other
            response items into the message's `content` field, rather than
            `additional_kwargs`. We recommend this format for new applications.

    ??? info "Structured output"

        ```python
        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(
                description="How funny the joke is, from 1 to 10"
            )


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup="Why was the cat sitting on the computer?",
            punchline="To keep an eye on the mouse!",
            rating=None,
        )
        ```

        See `with_structured_output` for more info.

    ??? info "JSON mode"

        ```python
        json_model = model.bind(response_format={"type": "json_object"})
        ai_msg = json_model.invoke(
            "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
        )
        ai_msg.content
        ```

        ```txt
        '\\n{\\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\\n}'
        ```

    ??? info "Image input"

        ```python
        import base64
        import httpx
        from langchain.messages import HumanMessage

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )

        ai_msg = model.invoke([message])
        ai_msg.content
        ```

        ```txt
        "The weather in the image appears to be clear and pleasant. The sky is mostly blue with scattered, light clouds, suggesting a sunny day with minimal cloud cover. There is no indication of rain or strong winds, and the overall scene looks bright and calm. The lush green grass and clear visibility further indicate good weather conditions."
        ```

    ??? info "Token usage"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata

        ```txt
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

        When streaming, set the `stream_usage` kwarg:

        ```python
        stream = model.stream(messages, stream_usage=True)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full.usage_metadata
        ```

        ```txt
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

    ??? info "Logprobs"

        ```python
        logprobs_model = model.bind(logprobs=True)
        ai_msg = logprobs_model.invoke(messages)
        ai_msg.response_metadata["logprobs"]
        ```

        ```txt
        {
            "content": [
                {
                    "token": "J",
                    "bytes": [74],
                    "logprob": -4.9617593e-06,
                    "top_logprobs": [],
                },
                {
                    "token": "'adore",
                    "bytes": [39, 97, 100, 111, 114, 101],
                    "logprob": -0.25202933,
                    "top_logprobs": [],
                },
                {
                    "token": " la",
                    "bytes": [32, 108, 97],
                    "logprob": -0.20141791,
                    "top_logprobs": [],
                },
                {
                    "token": " programmation",
                    "bytes": [
                        32,
                        112,
                        114,
                        111,
                        103,
                        114,
                        97,
                        109,
                        109,
                        97,
                        116,
                        105,
                        111,
                        110,
                    ],
                    "logprob": -1.9361265e-07,
                    "top_logprobs": [],
                },
                {
                    "token": ".",
                    "bytes": [46],
                    "logprob": -1.2233183e-05,
                    "top_logprobs": [],
                },
            ]
        }
        ```

    ??? info "Response metadata"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```

        ```txt
        {
            "token_usage": {
                "completion_tokens": 5,
                "prompt_tokens": 28,
                "total_tokens": 33,
            },
            "model_name": "gpt-4o",
            "system_fingerprint": "fp_319be4768e",
            "finish_reason": "stop",
            "logprobs": None,
        }
        ```

    ??? info "Flex processing"

        OpenAI offers a variety of [service tiers](https://platform.openai.com/docs/guides/flex-processing?api-mode=responses).
        The "flex" tier offers cheaper pricing for requests, with the trade-off that
        responses may take longer and resources might not always be available.
        This approach is best suited for non-critical tasks, including model testing,
        data enhancement, or jobs that can be run asynchronously.

        To use it, initialize the model with `service_tier="flex"`:

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="...", service_tier="flex")
        ```

        Note that this is a beta feature that is only available for a subset of models.
        See OpenAI [flex processing docs](https://platform.openai.com/docs/guides/flex-processing?api-mode=responses)
        for more detail.

    ??? info "OpenAI-compatible APIs"

        `ChatOpenAI` can be used with OpenAI-compatible APIs like
        [LM Studio](https://lmstudio.ai/), [vLLM](https://github.com/vllm-project/vllm),
        [Ollama](https://ollama.com/), and others.

        To use custom parameters specific to these providers, use the `extra_body` parameter.

        !!! example "LM Studio example with TTL (auto-eviction)"

            ```python
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",  # Can be any string
                model="mlx-community/QwQ-32B-4bit",
                temperature=0,
                extra_body={
                    "ttl": 300
                },  # Auto-evict model after 5 minutes of inactivity
            )
            ```

        !!! example "vLLM example with custom parameters"

            ```python
            model = ChatOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model="meta-llama/Llama-2-7b-chat-hf",
                extra_body={"use_beam_search": True, "best_of": 4},
            )
            ```

    ??? info "`model_kwargs` vs `extra_body`"

        Use the correct parameter for different types of API arguments:

        **Use `model_kwargs` for:**

        - Standard OpenAI API parameters not explicitly defined as class parameters
        - Parameters that should be flattened into the top-level request payload
        - Examples: `max_completion_tokens`, `stream_options`, `modalities`, `audio`

        ```python
        # Standard OpenAI parameters
        model = ChatOpenAI(
            model="...",
            model_kwargs={
                "stream_options": {"include_usage": True},
                "max_completion_tokens": 300,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "wav"},
            },
        )
        ```

        **Use `extra_body` for:**

        - Custom parameters specific to OpenAI-compatible providers (vLLM, LM Studio,
            OpenRouter, etc.)
        - Parameters that need to be nested under `extra_body` in the request
        - Any non-standard OpenAI API parameters

        ```python
        # Custom provider parameters
        model = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            model="custom-model",
            extra_body={
                "use_beam_search": True,  # vLLM parameter
                "best_of": 4,  # vLLM parameter
                "ttl": 300,  # LM Studio parameter
            },
        )
        ```

        **Key Differences:**

        - `model_kwargs`: Parameters are **merged into top-level** request payload
        - `extra_body`: Parameters are **nested under `extra_body`** key in request

        !!! warning
            Always use `extra_body` for custom parameters, **not** `model_kwargs`.
            Using `model_kwargs` for non-OpenAI parameters will cause API errors.

    ??? info "Prompt caching optimization"

        For high-volume applications with repetitive prompts, use `prompt_cache_key`
        per-invocation to improve cache hit rates and reduce costs:

        ```python
        model = ChatOpenAI(model="...")

        response = model.invoke(
            messages,
            prompt_cache_key="example-key-a",  # Routes to same machine for cache hits
        )

        customer_response = model.invoke(messages, prompt_cache_key="example-key-b")
        support_response = model.invoke(messages, prompt_cache_key="example-key-c")

        # Dynamic cache keys based on context
        cache_key = f"example-key-{dynamic_suffix}"
        response = model.invoke(messages, prompt_cache_key=cache_key)
        ```

        Cache keys help ensure requests with the same prompt prefix are routed to
        machines with existing cache, providing cost reduction and latency improvement on
        cached tokens.
    """  # noqa: E501

    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")
    """Maximum number of tokens to generate."""

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret environment variables."""
        return {"openai_api_key": "OPENAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "chat_models", "openai"]`
        """
        return ["langchain", "chat_models", "openai"]

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """Get the attributes of the langchain object."""
        attributes: dict[str, Any] = {}

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.openai_api_base:
            attributes["openai_api_base"] = self.openai_api_base

        if self.openai_proxy:
            attributes["openai_proxy"] = self.openai_proxy

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = super()._default_params
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        return params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # max_tokens was deprecated in favor of max_completion_tokens
        # in September 2024 release
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        # Mutate system message role to "developer" for o-series models
        if self.model_name and re.match(r"^o\d", self.model_name):
            for message in payload.get("messages", []):
                if message["role"] == "system":
                    message["role"] = "developer"
        return payload

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return super()._stream_responses(*args, **kwargs)
        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        tools: list | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a `TypedDict` class,
                - or a Pydantic class.

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.

            method: The method for steering model generation, one of:

                - `'json_schema'`:
                    Uses OpenAI's [Structured Output API](https://platform.openai.com/docs/guides/structured-outputs).
                    See the docs for [supported models](https://platform.openai.com/docs/guides/structured-outputs#supported-models).
                - `'function_calling'`:
                    Uses OpenAI's [tool-calling API](https://platform.openai.com/docs/guides/function-calling)
                    (formerly called function calling).
                - `'json_mode'`:
                    Uses OpenAI's [JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode).
                    Note that if using JSON mode then you must include instructions for
                    formatting the output into the desired schema into the model call.

                Learn more about the [differences between methods](https://platform.openai.com/docs/guides/structured-outputs#function-calling-vs-response-format).

            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:

                - `True`:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the
                    [supported schemas](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas).
                - `False`:
                    Input schema will not be validated and model output will not be
                    validated.
                - `None`:
                    `strict` argument will not be passed to the model.

                If schema is specified via `TypedDict` or JSON schema, `strict` is not
                enabled by default. Pass `strict=True` to enable it.

                !!! note
                    `strict` can only be non-null if `method` is `'json_schema'` or `'function_calling'`.
            tools:
                A list of tool-like objects to bind to the chat model. Requires that:

                - `method` is `'json_schema'` (default).
                - `strict=True`
                - `include_raw=True`

                If a model elects to call a
                tool, the resulting `AIMessage` in `'raw'` will include tool calls.

                ??? example

                    ```python
                    from langchain.chat_models import init_chat_model
                    from pydantic import BaseModel


                    class ResponseSchema(BaseModel):
                        response: str


                    def get_weather(location: str) -> str:
                        \"\"\"Get weather at a location.\"\"\"
                        pass

                    model = init_chat_model("openai:gpt-4o-mini")

                    structured_model = model.with_structured_output(
                        ResponseSchema,
                        tools=[get_weather],
                        strict=True,
                        include_raw=True,
                    )

                    structured_model.invoke("What's the weather in Boston?")
                    ```

                    ```python
                    {
                        "raw": AIMessage(content="", tool_calls=[...], ...),
                        "parsing_error": None,
                        "parsed": None,
                    }
                    ```

            kwargs: Additional keyword args are passed through to the model.

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

        !!! warning "Behavior changed in `langchain-openai` 0.3.0"

            `method` default changed from `"function_calling"` to `"json_schema"`.

        !!! warning "Behavior changed in `langchain-openai` 0.3.12"

            Support for `tools` added.

        !!! warning "Behavior changed in `langchain-openai` 0.3.21"

            Pass `kwargs` through to the model.

        ??? note "Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=False`, `strict=True`"

            Note, OpenAI has a number of restrictions on what types of schemas can be
            provided if `strict = True`. When using Pydantic, our model cannot
            specify any Field metadata (like min/max constraints) and fields cannot
            have default values.

            See [all constraints](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas).

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=..., description="A justification for the answer."
                )


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            AnswerWithJustification(
                answer="They weigh the same",
                justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
            )
            ```

        ??? note "Example: `schema=Pydantic` class, `method='function_calling'`, `include_raw=False`, `strict=False`"

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=..., description="A justification for the answer."
                )


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="function_calling"
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            AnswerWithJustification(
                answer="They weigh the same",
                justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
            )
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=True`"

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_Ao02pnFYXD6GN1yzc0uXPsvF",
                                "function": {
                                    "arguments": '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}',
                                    "name": "AnswerWithJustification",
                                },
                                "type": "function",
                            }
                        ]
                    },
                ),
                "parsed": AnswerWithJustification(
                    answer="They weigh the same.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=TypedDict` class, `method='json_schema'`, `include_raw=False`, `strict=False`"

            ```python
            from typing_extensions import Annotated, TypedDict

            from langchain_openai import ChatOpenAI


            class AnswerWithJustification(TypedDict):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: Annotated[
                    str | None, None, "A justification for the answer."
                ]


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "Example: `schema=OpenAI` function schema, `method='json_schema'`, `include_raw=False`"

            ```python
            from langchain_openai import ChatOpenAI

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

            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(oai_schema)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_mode'`, `include_raw=True`"

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                answer: str
                justification: str


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'
                ),
                "parsed": AnswerWithJustification(
                    answer="They are both the same weight.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=None`, `method='json_mode'`, `include_raw=True`"

            ```python
            structured_model = model.with_structured_output(
                method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'
                ),
                "parsed": {
                    "answer": "They are both the same weight.",
                    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                },
                "parsing_error": None,
            }
            ```

        """  # noqa: E501
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            tools=tools,
            **kwargs,
        )


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
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


def _url_to_size(image_source: str) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore[import]
    except ImportError:
        logger.info(
            "Unable to count image tokens. To count image tokens please install "
            "`pip install -U pillow httpx`."
        )
        return None
    if _is_url(image_source):
        try:
            import httpx
        except ImportError:
            logger.info(
                "Unable to count image tokens. To count image tokens please install "
                "`pip install -U httpx`."
            )
            return None
        response = httpx.get(image_source)
        response.raise_for_status()
        width, height = Image.open(BytesIO(response.content)).size
        return width, height
    if _is_b64(image_source):
        _, encoded = image_source.split(",", 1)
        data = base64.b64decode(encoded)
        width, height = Image.open(BytesIO(data)).size
        return width, height
    return None


def _count_image_tokens(width: int, height: int) -> int:
    # Reference: https://platform.openai.com/docs/guides/vision/calculating-costs
    width, height = _resize(width, height)
    h = ceil(height / 512)
    w = ceil(width / 512)
    return (170 * h * w) + 85


def _is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug("Unable to parse URL: %s", e)
        return False


def _is_b64(s: str) -> bool:
    return s.startswith("data:image")


def _resize(width: int, height: int) -> tuple[int, int]:
    # larger side must be <= 2048
    if width > 2048 or height > 2048:
        if width > height:
            height = (height * 2048) // width
            width = 2048
        else:
            width = (width * 2048) // height
            height = 2048
    # smaller side must be <= 768
    if width > 768 and height > 768:
        if width > height:
            width = (width * 768) // height
            height = 768
        else:
            height = (height * 768) // width
            width = 768
    return width, height


def _convert_to_openai_response_format(
    schema: dict[str, Any] | type, *, strict: bool | None = None
) -> dict | TypeBaseModel:
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return schema

    if (
        isinstance(schema, dict)
        and "json_schema" in schema
        and schema.get("type") == "json_schema"
    ):
        response_format = schema
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):
                strict = schema["strict"]
            else:
                strict = False
        function = convert_to_openai_function(schema, strict=strict)
        function["schema"] = function.pop("parameters")
        response_format = {"type": "json_schema", "json_schema": function}

    if (
        strict is not None
        and strict is not response_format["json_schema"].get("strict")
        and isinstance(schema, dict)
        and "strict" in schema.get("json_schema", {})
    ):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{schema['json_schema']['strict']} but 'strict' also passed in to "
            f"with_structured_output as {strict}. Please make sure that "
            f"'strict' is only specified in one place."
        )
        raise ValueError(msg)
    return response_format


def _oai_structured_outputs_parser(
    ai_msg: AIMessage, schema: type[_BM]
) -> PydanticBaseModel | None:
    if parsed := ai_msg.additional_kwargs.get("parsed"):
        if isinstance(parsed, dict):
            return schema(**parsed)
        return parsed
    if ai_msg.additional_kwargs.get("refusal"):
        raise OpenAIRefusalError(ai_msg.additional_kwargs["refusal"])
    if any(
        isinstance(block, dict)
        and block.get("type") == "non_standard"
        and "refusal" in block["value"]  # type: ignore[typeddict-item]
        for block in ai_msg.content_blocks
    ):
        refusal = next(
            block["value"]["refusal"]
            for block in ai_msg.content_blocks
            if isinstance(block, dict)
            and block["type"] == "non_standard"
            and "refusal" in block["value"]
        )
        raise OpenAIRefusalError(refusal)
    if ai_msg.tool_calls:
        return None
    msg = (
        "Structured Output response does not have a 'parsed' field nor a 'refusal' "
        f"field. Received message:\n\n{ai_msg}"
    )
    raise ValueError(msg)


class OpenAIRefusalError(Exception):
    """Error raised when OpenAI Structured Outputs API returns a refusal.

    When using OpenAI's Structured Outputs API with user-generated input, the model
    may occasionally refuse to fulfill the request for safety reasons.

    See [more on refusals](https://platform.openai.com/docs/guides/structured-outputs/refusals).
    """


def _create_usage_metadata(
    oai_token_usage: dict, service_tier: str | None = None
) -> UsageMetadata:
    input_tokens = oai_token_usage.get("prompt_tokens") or 0
    output_tokens = oai_token_usage.get("completion_tokens") or 0
    total_tokens = oai_token_usage.get("total_tokens") or input_tokens + output_tokens
    if service_tier not in {"priority", "flex"}:
        service_tier = None
    service_tier_prefix = f"{service_tier}_" if service_tier else ""
    input_token_details: dict = {
        "audio": (oai_token_usage.get("prompt_tokens_details") or {}).get(
            "audio_tokens"
        ),
        f"{service_tier_prefix}cache_read": (
            oai_token_usage.get("prompt_tokens_details") or {}
        ).get("cached_tokens"),
    }
    output_token_details: dict = {
        "audio": (oai_token_usage.get("completion_tokens_details") or {}).get(
            "audio_tokens"
        ),
        f"{service_tier_prefix}reasoning": (
            oai_token_usage.get("completion_tokens_details") or {}
        ).get("reasoning_tokens"),
    }
    if service_tier is not None:
        # Avoid counting cache and reasoning tokens towards the service tier token
        # counts, since service tier tokens are already priced differently
        input_token_details[service_tier] = input_tokens - input_token_details.get(
            f"{service_tier_prefix}cache_read", 0
        )
        output_token_details[service_tier] = output_tokens - output_token_details.get(
            f"{service_tier_prefix}reasoning", 0
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None}
        ),
        output_token_details=OutputTokenDetails(
            **{k: v for k, v in output_token_details.items() if v is not None}
        ),
    )


def _create_usage_metadata_responses(
    oai_token_usage: dict, service_tier: str | None = None
) -> UsageMetadata:
    input_tokens = oai_token_usage.get("input_tokens", 0)
    output_tokens = oai_token_usage.get("output_tokens", 0)
    total_tokens = oai_token_usage.get("total_tokens", input_tokens + output_tokens)
    if service_tier not in {"priority", "flex"}:
        service_tier = None
    service_tier_prefix = f"{service_tier}_" if service_tier else ""
    output_token_details: dict = {
        f"{service_tier_prefix}reasoning": (
            oai_token_usage.get("output_tokens_details") or {}
        ).get("reasoning_tokens")
    }
    input_token_details: dict = {
        f"{service_tier_prefix}cache_read": (
            oai_token_usage.get("input_tokens_details") or {}
        ).get("cached_tokens")
    }
    if service_tier is not None:
        # Avoid counting cache and reasoning tokens towards the service tier token
        # counts, since service tier tokens are already priced differently
        output_token_details[service_tier] = output_tokens - output_token_details.get(
            f"{service_tier_prefix}reasoning", 0
        )
        input_token_details[service_tier] = input_tokens - input_token_details.get(
            f"{service_tier_prefix}cache_read", 0
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None}
        ),
        output_token_details=OutputTokenDetails(
            **{k: v for k, v in output_token_details.items() if v is not None}
        ),
    )


def _is_builtin_tool(tool: dict) -> bool:
    return "type" in tool and tool["type"] != "function"


def _use_responses_api(payload: dict) -> bool:
    uses_builtin_tools = "tools" in payload and any(
        _is_builtin_tool(tool) for tool in payload["tools"]
    )
    responses_only_args = {
        "include",
        "previous_response_id",
        "reasoning",
        "text",
        "truncation",
    }
    return bool(uses_builtin_tools or responses_only_args.intersection(payload))


def _get_last_messages(
    messages: Sequence[BaseMessage],
) -> tuple[Sequence[BaseMessage], str | None]:
    """Get the last part of the conversation after the last `AIMessage` with an `id`.

    Will return:

    1. Every message after the most-recent `AIMessage` that has a non-empty
        `response_metadata["id"]` (may be an empty list),
    2. That `id`.

    If the most-recent `AIMessage` does not have an `id` (or there is no
    `AIMessage` at all) the entire conversation is returned together with `None`.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            response_id = msg.response_metadata.get("id")
            if response_id and response_id.startswith("resp_"):
                return messages[i + 1 :], response_id
            # Continue searching for an AIMessage with a valid response_id

    return messages, None


def _construct_responses_api_payload(
    messages: Sequence[BaseMessage], payload: dict
) -> dict:
    # Rename legacy parameters
    for legacy_token_param in ["max_tokens", "max_completion_tokens"]:
        if legacy_token_param in payload:
            payload["max_output_tokens"] = payload.pop(legacy_token_param)
    if "reasoning_effort" in payload and "reasoning" not in payload:
        payload["reasoning"] = {"effort": payload.pop("reasoning_effort")}

    # Remove temperature parameter for models that don't support it in responses API
    # gpt-5-chat supports temperature, and gpt-5 models with reasoning.effort='none'
    # also support temperature
    model = payload.get("model") or ""
    if (
        model.startswith("gpt-5")
        and ("chat" not in model)  # gpt-5-chat supports
        and (payload.get("reasoning") or {}).get("effort") != "none"
    ):
        payload.pop("temperature", None)

    payload["input"] = _construct_responses_api_input(messages)
    if tools := payload.pop("tools", None):
        new_tools: list = []
        for tool in tools:
            # chat api: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}, "strict": ...}}  # noqa: E501
            # responses api: {"type": "function", "name": "...", "description": "...", "parameters": {...}, "strict": ...}  # noqa: E501
            if tool["type"] == "function" and "function" in tool:
                new_tools.append({"type": "function", **tool["function"]})
            else:
                if tool["type"] == "image_generation":
                    # Handle partial images (not yet supported)
                    if "partial_images" in tool:
                        msg = (
                            "Partial image generation is not yet supported "
                            "via the LangChain ChatOpenAI client. Please "
                            "drop the 'partial_images' key from the image_generation "
                            "tool."
                        )
                        raise NotImplementedError(msg)
                    if payload.get("stream") and "partial_images" not in tool:
                        # OpenAI requires this parameter be set; we ignore it during
                        # streaming.
                        tool = {**tool, "partial_images": 1}
                    else:
                        pass

                new_tools.append(tool)

        payload["tools"] = new_tools
    if tool_choice := payload.pop("tool_choice", None):
        # chat api: {"type": "function", "function": {"name": "..."}}
        # responses api: {"type": "function", "name": "..."}
        if (
            isinstance(tool_choice, dict)
            and tool_choice["type"] == "function"
            and "function" in tool_choice
        ):
            payload["tool_choice"] = {"type": "function", **tool_choice["function"]}
        else:
            payload["tool_choice"] = tool_choice

    # Structured output
    if schema := payload.pop("response_format", None):
        # For pydantic + non-streaming case, we use responses.parse.
        # Otherwise, we use responses.create.
        strict = payload.pop("strict", None)
        if not payload.get("stream") and _is_pydantic_class(schema):
            payload["text_format"] = schema
        else:
            if _is_pydantic_class(schema):
                schema_dict = schema.model_json_schema()
                strict = True
            else:
                schema_dict = schema
            if schema_dict == {"type": "json_object"}:  # JSON mode
                if "text" in payload and isinstance(payload["text"], dict):
                    payload["text"]["format"] = {"type": "json_object"}
                else:
                    payload["text"] = {"format": {"type": "json_object"}}
            elif (
                (
                    response_format := _convert_to_openai_response_format(
                        schema_dict, strict=strict
                    )
                )
                and (isinstance(response_format, dict))
                and (response_format["type"] == "json_schema")
            ):
                format_value = {"type": "json_schema", **response_format["json_schema"]}
                if "text" in payload and isinstance(payload["text"], dict):
                    payload["text"]["format"] = format_value
                else:
                    payload["text"] = {"format": format_value}
            else:
                pass

    verbosity = payload.pop("verbosity", None)
    if verbosity is not None:
        if "text" in payload and isinstance(payload["text"], dict):
            payload["text"]["verbosity"] = verbosity
        else:
            payload["text"] = {"verbosity": verbosity}

    return payload


def _format_annotation_to_lc(annotation: dict[str, Any]) -> dict[str, Any]:
    # langchain-core reserves the `"index"` key for streaming aggregation.
    # Here we re-name.
    if annotation.get("type") == "file_citation" and "index" in annotation:
        new_annotation = annotation.copy()
        new_annotation["file_index"] = new_annotation.pop("index")
        return new_annotation
    return annotation


def _format_annotation_from_lc(annotation: dict[str, Any]) -> dict[str, Any]:
    if annotation.get("type") == "file_citation" and "file_index" in annotation:
        new_annotation = annotation.copy()
        new_annotation["index"] = new_annotation.pop("file_index")
        return new_annotation
    return annotation


def _convert_chat_completions_blocks_to_responses(
    block: dict[str, Any],
) -> dict[str, Any]:
    """Convert chat completions content blocks to Responses API format.

    Only handles text, image, file blocks. Others pass through.
    """
    if block["type"] == "text":
        # chat api: {"type": "text", "text": "..."}
        # responses api: {"type": "input_text", "text": "..."}
        return {"type": "input_text", "text": block["text"]}
    if block["type"] == "image_url":
        # chat api: {"type": "image_url", "image_url": {"url": "...", "detail": "..."}}  # noqa: E501
        # responses api: {"type": "image_url", "image_url": "...", "detail": "...", "file_id": "..."}  # noqa: E501
        new_block = {
            "type": "input_image",
            "image_url": block["image_url"]["url"],
        }
        if block["image_url"].get("detail"):
            new_block["detail"] = block["image_url"]["detail"]
        return new_block
    if block["type"] == "file":
        return {"type": "input_file", **block["file"]}
    return block


def _ensure_valid_tool_message_content(tool_output: Any) -> str | list[dict]:
    if isinstance(tool_output, str):
        return tool_output
    if isinstance(tool_output, list) and all(
        isinstance(block, dict)
        and block.get("type")
        in (
            "input_text",
            "input_image",
            "input_file",
            "text",
            "image_url",
            "file",
        )
        for block in tool_output
    ):
        return [
            _convert_chat_completions_blocks_to_responses(block)
            for block in tool_output
        ]
    return _stringify(tool_output)


def _make_computer_call_output_from_message(
    message: ToolMessage,
) -> dict[str, Any] | None:
    computer_call_output: dict[str, Any] | None = None
    if isinstance(message.content, list):
        for block in message.content:
            if (
                message.additional_kwargs.get("type") == "computer_call_output"
                and isinstance(block, dict)
                and block.get("type") == "input_image"
            ):
                # Use first input_image block
                computer_call_output = {
                    "call_id": message.tool_call_id,
                    "type": "computer_call_output",
                    "output": block,
                }
                break
            if (
                isinstance(block, dict)
                and block.get("type") == "non_standard"
                and block.get("value", {}).get("type") == "computer_call_output"
            ):
                computer_call_output = block["value"]
                break
    elif message.additional_kwargs.get("type") == "computer_call_output":
        # string, assume image_url
        computer_call_output = {
            "call_id": message.tool_call_id,
            "type": "computer_call_output",
            "output": {"type": "input_image", "image_url": message.content},
        }
    if (
        computer_call_output is not None
        and "acknowledged_safety_checks" in message.additional_kwargs
    ):
        computer_call_output["acknowledged_safety_checks"] = message.additional_kwargs[
            "acknowledged_safety_checks"
        ]
    return computer_call_output


def _make_custom_tool_output_from_message(message: ToolMessage) -> dict | None:
    custom_tool_output = None
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "custom_tool_call_output":
            custom_tool_output = {
                "type": "custom_tool_call_output",
                "call_id": message.tool_call_id,
                "output": block.get("output") or "",
            }
            break
        if (
            isinstance(block, dict)
            and block.get("type") == "non_standard"
            and block.get("value", {}).get("type") == "custom_tool_call_output"
        ):
            custom_tool_output = block["value"]
            break

    return custom_tool_output


def _pop_index_and_sub_index(block: dict) -> dict:
    """When streaming, `langchain-core` uses `index` to aggregate text blocks.

    OpenAI API does not support this key, so we need to remove it.
    """
    new_block = {k: v for k, v in block.items() if k != "index"}
    if "summary" in new_block and isinstance(new_block["summary"], list):
        new_summary = []
        for sub_block in new_block["summary"]:
            new_sub_block = {k: v for k, v in sub_block.items() if k != "index"}
            new_summary.append(new_sub_block)
        new_block["summary"] = new_summary
    return new_block


def _construct_responses_api_input(messages: Sequence[BaseMessage]) -> list:
    """Construct the input for the OpenAI Responses API."""
    input_ = []
    for lc_msg in messages:
        if isinstance(lc_msg, AIMessage):
            lc_msg = _convert_from_v03_ai_message(lc_msg)
            msg = _convert_message_to_dict(lc_msg, api="responses")
            if isinstance(msg.get("content"), list) and all(
                isinstance(block, dict) for block in msg["content"]
            ):
                tcs: list[types.ToolCall] = [
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call.get("id"),
                    }
                    for tool_call in lc_msg.tool_calls
                ]
                msg["content"] = _convert_from_v1_to_responses(msg["content"], tcs)
        else:
            msg = _convert_message_to_dict(lc_msg, api="responses")
            # Get content from non-standard content blocks
            if isinstance(msg["content"], list):
                for i, block in enumerate(msg["content"]):
                    if isinstance(block, dict) and block.get("type") == "non_standard":
                        msg["content"][i] = block["value"]
        # "name" parameter unsupported
        if "name" in msg:
            msg.pop("name")
        if msg["role"] == "tool":
            tool_output = msg["content"]
            computer_call_output = _make_computer_call_output_from_message(
                cast(ToolMessage, lc_msg)
            )
            custom_tool_output = _make_custom_tool_output_from_message(lc_msg)  # type: ignore[arg-type]
            if computer_call_output:
                input_.append(computer_call_output)
            elif custom_tool_output:
                input_.append(custom_tool_output)
            else:
                tool_output = _ensure_valid_tool_message_content(tool_output)
                function_call_output = {
                    "type": "function_call_output",
                    "output": tool_output,
                    "call_id": msg["tool_call_id"],
                }
                input_.append(function_call_output)
        elif msg["role"] == "assistant":
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and (block_type := block.get("type")):
                        # Aggregate content blocks for a single message
                        if block_type in ("text", "output_text", "refusal"):
                            msg_id = block.get("id")
                            if block_type in ("text", "output_text"):
                                # Defensive check: block may not have "text" key
                                text = block.get("text")
                                if text is None:
                                    # Skip blocks without text content
                                    continue
                                new_block = {
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [
                                        _format_annotation_from_lc(annotation)
                                        for annotation in block.get("annotations") or []
                                    ],
                                }
                            elif block_type == "refusal":
                                new_block = {
                                    "type": "refusal",
                                    "refusal": block["refusal"],
                                }
                            for item in input_:
                                if (item_id := item.get("id")) and item_id == msg_id:
                                    # If existing block with this ID, append to it
                                    if "content" not in item:
                                        item["content"] = []
                                    item["content"].append(new_block)
                                    break
                            else:
                                # If no block with this ID, create a new one
                                input_.append(
                                    {
                                        "type": "message",
                                        "content": [new_block],
                                        "role": "assistant",
                                        "id": msg_id,
                                    }
                                )
                        elif block_type in (
                            "reasoning",
                            "web_search_call",
                            "file_search_call",
                            "function_call",
                            "computer_call",
                            "custom_tool_call",
                            "code_interpreter_call",
                            "mcp_call",
                            "mcp_list_tools",
                            "mcp_approval_request",
                        ):
                            input_.append(_pop_index_and_sub_index(block))
                        elif block_type == "image_generation_call":
                            # A previous image generation call can be referenced by ID
                            input_.append(
                                {"type": "image_generation_call", "id": block["id"]}
                            )
                        else:
                            pass
            elif isinstance(msg.get("content"), str):
                input_.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": msg["content"],
                                "annotations": [],
                            }
                        ],
                    }
                )

            # Add function calls from tool calls if not already present
            if tool_calls := msg.pop("tool_calls", None):
                content_call_ids = {
                    block["call_id"]
                    for block in input_
                    if block.get("type") in ("function_call", "custom_tool_call")
                    and "call_id" in block
                }
                for tool_call in tool_calls:
                    if tool_call["id"] not in content_call_ids:
                        function_call = {
                            "type": "function_call",
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                            "call_id": tool_call["id"],
                        }
                        input_.append(function_call)

        elif msg["role"] in ("user", "system", "developer"):
            if isinstance(msg["content"], list):
                new_blocks = []
                non_message_item_types = ("mcp_approval_response",)
                for block in msg["content"]:
                    if block["type"] in ("text", "image_url", "file"):
                        new_blocks.append(
                            _convert_chat_completions_blocks_to_responses(block)
                        )
                    elif block["type"] in ("input_text", "input_image", "input_file"):
                        new_blocks.append(block)
                    elif block["type"] in non_message_item_types:
                        input_.append(block)
                    else:
                        pass
                msg["content"] = new_blocks
                if msg["content"]:
                    input_.append(msg)
            else:
                input_.append(msg)
        else:
            input_.append(msg)

    return input_


def _get_output_text(response: Response) -> str:
    """Safe output text extraction.

    Context: OpenAI SDK deleted `response.output_text` momentarily in `1.99.2`.
    """
    if hasattr(response, "output_text"):
        return response.output_text
    texts = [
        content.text
        for output in response.output
        if output.type == "message"
        for content in output.content
        if content.type == "output_text"
    ]
    return "".join(texts)


def _construct_lc_result_from_responses_api(
    response: Response,
    schema: type[_BM] | None = None,
    metadata: dict | None = None,
    output_version: str | None = None,
) -> ChatResult:
    """Construct `ChatResponse` from OpenAI Response API response."""
    if response.error:
        raise ValueError(response.error)

    if output_version is None:
        # Sentinel value of None lets us know if output_version is set explicitly.
        # Explicitly setting `output_version="responses/v1"` separately enables the
        # Responses API.
        output_version = "responses/v1"

    response_metadata = {
        k: v
        for k, v in response.model_dump(exclude_none=True, mode="json").items()
        if k
        in (
            "created_at",
            # backwards compatibility: keep response ID in response_metadata as well as
            # top-level-id
            "id",
            "incomplete_details",
            "metadata",
            "object",
            "status",
            "user",
            "model",
            "service_tier",
        )
    }
    if metadata:
        response_metadata.update(metadata)
    # for compatibility with chat completion calls.
    response_metadata["model_provider"] = "openai"
    response_metadata["model_name"] = response_metadata.get("model")
    if response.usage:
        usage_metadata = _create_usage_metadata_responses(
            response.usage.model_dump(), response.service_tier
        )
    else:
        usage_metadata = None

    content_blocks: list = []
    tool_calls = []
    invalid_tool_calls = []
    additional_kwargs: dict = {}
    for output in response.output:
        if output.type == "message":
            for content in output.content:
                if content.type == "output_text":
                    block = {
                        "type": "text",
                        "text": content.text,
                        "annotations": [
                            _format_annotation_to_lc(annotation.model_dump())
                            for annotation in content.annotations
                        ]
                        if isinstance(content.annotations, list)
                        else [],
                        "id": output.id,
                    }
                    content_blocks.append(block)
                    if hasattr(content, "parsed"):
                        additional_kwargs["parsed"] = content.parsed
                if content.type == "refusal":
                    content_blocks.append(
                        {"type": "refusal", "refusal": content.refusal, "id": output.id}
                    )
        elif output.type == "function_call":
            content_blocks.append(output.model_dump(exclude_none=True, mode="json"))
            try:
                args = json.loads(output.arguments, strict=False)
                error = None
            except JSONDecodeError as e:
                args = output.arguments
                error = str(e)
            if error is None:
                tool_call = {
                    "type": "tool_call",
                    "name": output.name,
                    "args": args,
                    "id": output.call_id,
                }
                tool_calls.append(tool_call)
            else:
                tool_call = {
                    "type": "invalid_tool_call",
                    "name": output.name,
                    "args": args,
                    "id": output.call_id,
                    "error": error,
                }
                invalid_tool_calls.append(tool_call)
        elif output.type == "custom_tool_call":
            content_blocks.append(output.model_dump(exclude_none=True, mode="json"))
            tool_call = {
                "type": "tool_call",
                "name": output.name,
                "args": {"__arg1": output.input},
                "id": output.call_id,
            }
            tool_calls.append(tool_call)
        elif output.type in (
            "reasoning",
            "web_search_call",
            "file_search_call",
            "computer_call",
            "code_interpreter_call",
            "mcp_call",
            "mcp_list_tools",
            "mcp_approval_request",
            "image_generation_call",
        ):
            content_blocks.append(output.model_dump(exclude_none=True, mode="json"))

    # Workaround for parsing structured output in the streaming case.
    #    from openai import OpenAI
    #    from pydantic import BaseModel

    #    class Foo(BaseModel):
    #        response: str

    #    client = OpenAI()

    #    client.responses.parse(
    #        model="...",
    #        input=[{"content": "how are ya", "role": "user"}],
    #        text_format=Foo,
    #        stream=True,  # <-- errors
    #    )
    output_text = _get_output_text(response)
    if (
        schema is not None
        and "parsed" not in additional_kwargs
        and output_text  # tool calls can generate empty output text
        and response.text
        and (text_config := response.text.model_dump())
        and (format_ := text_config.get("format", {}))
        and (format_.get("type") == "json_schema")
    ):
        try:
            parsed_dict = json.loads(output_text)
            if schema and _is_pydantic_class(schema):
                parsed = schema(**parsed_dict)
            else:
                parsed = parsed_dict
            additional_kwargs["parsed"] = parsed
        except json.JSONDecodeError:
            pass

    message = AIMessage(
        content=content_blocks,
        id=response.id,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )
    if output_version == "v0":
        message = _convert_to_v03_ai_message(message)

    return ChatResult(generations=[ChatGeneration(message=message)])


def _convert_responses_chunk_to_generation_chunk(
    chunk: Any,
    current_index: int,  # index in content
    current_output_index: int,  # index in Response output
    current_sub_index: int,  # index of content block in output item
    schema: type[_BM] | None = None,
    metadata: dict | None = None,
    has_reasoning: bool = False,
    output_version: str | None = None,
) -> tuple[int, int, int, ChatGenerationChunk | None]:
    def _advance(output_idx: int, sub_idx: int | None = None) -> None:
        """Advance indexes tracked during streaming.

        Example: we stream a response item of the form:

        ```python
        {
            "type": "message",  # output_index 0
            "role": "assistant",
            "id": "msg_123",
            "content": [
                {"type": "output_text", "text": "foo"},  # sub_index 0
                {"type": "output_text", "text": "bar"},  # sub_index 1
            ],
        }
        ```

        This is a single item with a shared `output_index` and two sub-indexes, one
        for each content block.

        This will be processed into an `AIMessage` with two text blocks:

        ```python
        AIMessage(
            [
                {"type": "text", "text": "foo", "id": "msg_123"},  # index 0
                {"type": "text", "text": "bar", "id": "msg_123"},  # index 1
            ]
        )
        ```

        This function just identifies updates in output or sub-indexes and increments
        the current index accordingly.
        """
        nonlocal current_index, current_output_index, current_sub_index
        if sub_idx is None:
            if current_output_index != output_idx:
                current_index += 1
        else:
            if (current_output_index != output_idx) or (current_sub_index != sub_idx):
                current_index += 1
            current_sub_index = sub_idx
        current_output_index = output_idx

    if output_version is None:
        # Sentinel value of None lets us know if output_version is set explicitly.
        # Explicitly setting `output_version="responses/v1"` separately enables the
        # Responses API.
        output_version = "responses/v1"

    content = []
    tool_call_chunks: list = []
    additional_kwargs: dict = {}
    response_metadata = metadata or {}
    response_metadata["model_provider"] = "openai"
    usage_metadata = None
    chunk_position: Literal["last"] | None = None
    id = None
    if chunk.type == "response.output_text.delta":
        _advance(chunk.output_index, chunk.content_index)
        content.append({"type": "text", "text": chunk.delta, "index": current_index})
    elif chunk.type == "response.output_text.annotation.added":
        _advance(chunk.output_index, chunk.content_index)
        if isinstance(chunk.annotation, dict):
            # Appears to be a breaking change in openai==1.82.0
            annotation = chunk.annotation
        else:
            annotation = chunk.annotation.model_dump(exclude_none=True, mode="json")

        content.append(
            {
                "type": "text",
                "annotations": [_format_annotation_to_lc(annotation)],
                "index": current_index,
            }
        )
    elif chunk.type == "response.output_text.done":
        _advance(chunk.output_index, chunk.content_index)
        content.append(
            {
                "type": "text",
                "text": "",
                "id": chunk.item_id,
                "index": current_index,
            }
        )
    elif chunk.type == "response.created":
        id = chunk.response.id
        response_metadata["id"] = chunk.response.id  # Backwards compatibility
    elif chunk.type in ("response.completed", "response.incomplete"):
        msg = cast(
            AIMessage,
            (
                _construct_lc_result_from_responses_api(
                    chunk.response, schema=schema, output_version=output_version
                )
                .generations[0]
                .message
            ),
        )
        if parsed := msg.additional_kwargs.get("parsed"):
            additional_kwargs["parsed"] = parsed
        usage_metadata = msg.usage_metadata
        response_metadata = {
            k: v for k, v in msg.response_metadata.items() if k != "id"
        }
        chunk_position = "last"
    elif chunk.type == "response.output_item.added" and chunk.item.type == "message":
        if output_version == "v0":
            id = chunk.item.id
        else:
            pass
    elif (
        chunk.type == "response.output_item.added"
        and chunk.item.type == "function_call"
    ):
        _advance(chunk.output_index)
        tool_call_chunks.append(
            {
                "type": "tool_call_chunk",
                "name": chunk.item.name,
                "args": chunk.item.arguments,
                "id": chunk.item.call_id,
                "index": current_index,
            }
        )
        content.append(
            {
                "type": "function_call",
                "name": chunk.item.name,
                "arguments": chunk.item.arguments,
                "call_id": chunk.item.call_id,
                "id": chunk.item.id,
                "index": current_index,
            }
        )
    elif chunk.type == "response.output_item.done" and chunk.item.type in (
        "web_search_call",
        "file_search_call",
        "computer_call",
        "code_interpreter_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "image_generation_call",
    ):
        _advance(chunk.output_index)
        tool_output = chunk.item.model_dump(exclude_none=True, mode="json")
        tool_output["index"] = current_index
        content.append(tool_output)
    elif (
        chunk.type == "response.output_item.done"
        and chunk.item.type == "custom_tool_call"
    ):
        _advance(chunk.output_index)
        tool_output = chunk.item.model_dump(exclude_none=True, mode="json")
        tool_output["index"] = current_index
        content.append(tool_output)
        tool_call_chunks.append(
            {
                "type": "tool_call_chunk",
                "name": chunk.item.name,
                "args": json.dumps({"__arg1": chunk.item.input}),
                "id": chunk.item.call_id,
                "index": current_index,
            }
        )
    elif chunk.type == "response.function_call_arguments.delta":
        _advance(chunk.output_index)
        tool_call_chunks.append(
            {"type": "tool_call_chunk", "args": chunk.delta, "index": current_index}
        )
        content.append(
            {"type": "function_call", "arguments": chunk.delta, "index": current_index}
        )
    elif chunk.type == "response.refusal.done":
        content.append({"type": "refusal", "refusal": chunk.refusal})
    elif chunk.type == "response.output_item.added" and chunk.item.type == "reasoning":
        _advance(chunk.output_index)
        current_sub_index = 0
        reasoning = chunk.item.model_dump(exclude_none=True, mode="json")
        reasoning["index"] = current_index
        content.append(reasoning)
    elif chunk.type == "response.reasoning_summary_part.added":
        _advance(chunk.output_index)
        content.append(
            {
                # langchain-core uses the `index` key to aggregate text blocks.
                "summary": [
                    {"index": chunk.summary_index, "type": "summary_text", "text": ""}
                ],
                "index": current_index,
                "type": "reasoning",
                "id": chunk.item_id,
            }
        )
    elif chunk.type == "response.image_generation_call.partial_image":
        # Partial images are not supported yet.
        pass
    elif chunk.type == "response.reasoning_summary_text.delta":
        _advance(chunk.output_index)
        content.append(
            {
                "summary": [
                    {
                        "index": chunk.summary_index,
                        "type": "summary_text",
                        "text": chunk.delta,
                    }
                ],
                "index": current_index,
                "type": "reasoning",
            }
        )
    else:
        return current_index, current_output_index, current_sub_index, None

    message = AIMessageChunk(
        content=content,  # type: ignore[arg-type]
        tool_call_chunks=tool_call_chunks,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
        id=id,
        chunk_position=chunk_position,
    )
    if output_version == "v0":
        message = cast(
            AIMessageChunk,
            _convert_to_v03_ai_message(message, has_reasoning=has_reasoning),
        )

    return (
        current_index,
        current_output_index,
        current_sub_index,
        ChatGenerationChunk(message=message),
    )
