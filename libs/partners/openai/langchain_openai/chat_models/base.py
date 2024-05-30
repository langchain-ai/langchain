"""OpenAI chat wrapper."""

from __future__ import annotations

import json
import logging
import os
import sys
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

import openai
import tiktoken
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
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
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.utils import build_extra_kwargs

logger = logging.getLogger(__name__)


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
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
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
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""), name=name, id=id_)
    elif role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    elif role == "tool":
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
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        # Remove unexpected block types
        formatted_content = []
        for block in content:
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] == "tool_use"
            ):
                continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {
        "content": _format_message_content(message.content),
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
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
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
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                {
                    "name": rtc["function"].get("name"),
                    "args": rtc["function"].get("arguments"),
                    "id": rtc.get("id"),
                    "index": rtc["index"],
                }
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    else:
        return default_class(content=content, id=id_)  # type: ignore


class _FunctionCall(TypedDict):
    name: str


_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]
_DictOrPydantic = Union[Dict, _BM]


class _AllReturnType(TypedDict):
    raw: BaseMessage
    parsed: Optional[_DictOrPydantic]
    parsing_error: Optional[BaseException]


class BaseChatOpenAI(BaseChatModel):
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or 
        None."""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    tiktoken_model_name: Optional[str] = None
    """The model name to pass to tiktoken when using this class. 
    Tiktoken is used to count the number of tokens in documents to constrain 
    them to be under a certain limit. By default, when set to None, this will 
    be the same as the embedding model name. However, there are some cases 
    where you may want to use this Embedding class with a model name not 
    supported by tiktoken. This can include when using Azure embeddings or 
    when using one of the many model providers that expose an OpenAI-like 
    API but with different models. In those cases, in order to avoid erroring 
    when tiktoken is called, you can specify a model name to use here."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client. Only used for sync invocations. Must specify 
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify 
        http_client as well if you'd like a custom client for sync invocations."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        values["openai_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "openai_api_key", "OPENAI_API_KEY")
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )

        client_params = {
            "api_key": (
                values["openai_api_key"].get_secret_value()
                if values["openai_api_key"]
                else None
            ),
            "organization": values["openai_organization"],
            "base_url": values["openai_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }

        openai_proxy = values["openai_proxy"]
        if not values.get("client"):
            if openai_proxy and not values["http_client"]:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                values["http_client"] = httpx.Client(proxy=openai_proxy)
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not values.get("async_client"):
            if openai_proxy and not values["http_async_client"]:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                values["http_async_client"] = httpx.AsyncClient(proxy=openai_proxy)
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        with self.client.create(messages=message_dicts, **params) as response:
            for chunk in response:
                if not isinstance(chunk, dict):
                    chunk = chunk.model_dump()
                if len(chunk["choices"]) == 0:
                    if token_usage := chunk.get("usage"):
                        usage_metadata = UsageMetadata(
                            input_tokens=token_usage.get("prompt_tokens", 0),
                            output_tokens=token_usage.get("completion_tokens", 0),
                            total_tokens=token_usage.get("total_tokens", 0),
                        )
                        chunk = ChatGenerationChunk(
                            message=default_chunk_class(
                                content="", usage_metadata=usage_metadata
                            )
                        )
                    else:
                        continue
                else:
                    choice = chunk["choices"][0]
                    if choice["delta"] is None:
                        continue
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], default_chunk_class
                    )
                    generation_info = {}
                    if finish_reason := choice.get("finish_reason"):
                        generation_info["finish_reason"] = finish_reason
                    logprobs = choice.get("logprobs")
                    if logprobs:
                        generation_info["logprobs"] = logprobs
                    default_chunk_class = chunk.__class__
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info or None
                    )
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text, chunk=chunk, logprobs=logprobs
                    )
                yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.create(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(
        self, response: Union[dict, openai.BaseModel]
    ) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()

        # Sometimes the AI Model calling will get error, we should raise it.
        # Otherwise, the next code 'choices.extend(response["choices"])'
        # will throw a "TypeError: 'NoneType' object is not iterable" error
        # to mask the true error. Because 'response["choices"]' is None.
        if response.get("error"):
            raise ValueError(response.get("error"))

        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        response = await self.async_client.create(messages=message_dicts, **params)
        async with response:
            async for chunk in response:
                if not isinstance(chunk, dict):
                    chunk = chunk.model_dump()
                if len(chunk["choices"]) == 0:
                    if token_usage := chunk.get("usage"):
                        usage_metadata = UsageMetadata(
                            input_tokens=token_usage.get("prompt_tokens", 0),
                            output_tokens=token_usage.get("completion_tokens", 0),
                            total_tokens=token_usage.get("total_tokens", 0),
                        )
                        chunk = ChatGenerationChunk(
                            message=default_chunk_class(
                                content="", usage_metadata=usage_metadata
                            )
                        )
                    else:
                        continue
                else:
                    choice = chunk["choices"][0]
                    if choice["delta"] is None:
                        continue
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], default_chunk_class
                    )
                    generation_info = {}
                    if finish_reason := choice.get("finish_reason"):
                        generation_info["finish_reason"] = finish_reason
                    logprobs = choice.get("logprobs")
                    if logprobs:
                        generation_info["logprobs"] = logprobs
                    default_chunk_class = chunk.__class__
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info or None
                    )
                if run_manager:
                    await run_manager.on_llm_new_token(
                        token=chunk.text, chunk=chunk, logprobs=logprobs
                    )
                yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await self.async_client.create(messages=message_dicts, **params)
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="openai",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai-chat"

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        return model, encoding

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text with tiktoken package."""
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("gpt-3.5-turbo-0301"):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}. See "
                "https://platform.openai.com/docs/guides/text-generation/managing-tokens"
                " for information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        function_call: Optional[
            Union[_FunctionCall, str, Literal["auto", "none"]]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind functions (and other objects) to this chat model.

        Assumes model is compatible with OpenAI function-calling API.

        NOTE: Using bind_tools is recommended instead, as the `functions` and
            `function_call` request parameters are officially marked as deprecated by
            OpenAI.

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any).
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_functions = [convert_to_openai_function(fn) for fn in functions]
        if function_call is not None:
            function_call = (
                {"name": function_call}
                if isinstance(function_call, str)
                and function_call not in ("auto", "none")
                else function_call
            )
            if isinstance(function_call, dict) and len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if (
                isinstance(function_call, dict)
                and formatted_functions[0]["name"] != function_call["name"]
            ):
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            kwargs = {**kwargs, "function_call": function_call}
        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Options are:
                name of the tool (str): calls corresponding tool;
                "auto": automatically selects a tool (including no tool);
                "none": does not call a tool;
                "any" or "required": force at least one tool to be called;
                True: forces tool call (requires `tools` be length 1);
                False: no effect;

                or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be used when a single tool is "
                        f"passed in, received {len(tools)} tools."
                    )
                tool_choice = {
                    "type": "function",
                    "function": {"name": formatted_tools[0]["function"]["name"]},
                }
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]:
        ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec or be a valid JSON schema
                with top level 'title' and 'description' keys specified.
            method: The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" then OpenAI's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_openai import ChatOpenAI
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_openai import ChatOpenAI
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_openai import ChatOpenAI
                from langchain_core.pydantic_v1 import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: JSON mode, Pydantic schema (method="json_mode", include_raw=True):
            .. code-block::

                from langchain_openai import ChatOpenAI
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                #     'parsing_error': None
                # }

        Example: JSON mode, no schema (schema=None, method="json_mode", include_raw=True):
            .. code-block::

                from langchain_openai import ChatOpenAI

                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': {
                #         'answer': 'They are both the same weight.',
                #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
                #     },
                #     'parsing_error': None
                # }


        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            llm = self.bind_tools([schema], tool_choice=True)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_format'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser


class ChatOpenAI(BaseChatOpenAI):
    """OpenAI chat model integration.

    Setup:
        Install ``langchain-openai`` and set environment variable ``OPENAI_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-openai
            export OPENAI_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of OpenAI model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        logprobs: Optional[bool]
            Whether to return logprobs.
        stream_options: Dict
            Configure streaming outputs, like whether to return token usage when
            streaming (``{"include_usage": True}``).

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            OpenAI API key. If not passed in will be read from env var OPENAI_API_KEY.
        base_url:
            Base URL for PAI requests. Only specify if using a proxy or service
            emulator.
        organization:
            OpenAI organization ID. If not passed in will be read from env
            var OPENAI_ORG_ID.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # base_url="...",
                # organization="...",
                # other params...
            )

    **NOTE**: Any param which is not explicitly supported will be passed directly to the
    ``openai.OpenAI.chat.completions.create(...)`` API every time to the model is
    invoked. For example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            import openai

            ChatOpenAI(..., frequency_penalty=0.2).invoke(...)

            # results in underlying API call of:

            openai.OpenAI(..).chat.completions.create(..., frequency_penalty=0.2)

            # which is also equivalent to:

            ChatOpenAI(...).invoke(..., frequency_penalty=0.2)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content="J'adore la programmation.", response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 31, 'total_tokens': 36}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_43dfabdef1', 'finish_reason': 'stop', 'logprobs': None}, id='run-012cffe2-5d3d-424d-83b5-51c6d4a593d1-0', usage_metadata={'input_tokens': 31, 'output_tokens': 5, 'total_tokens': 36})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content='', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='J', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content="'adore", id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content=' la', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content=' programmation', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='.', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='', response_metadata={'finish_reason': 'stop'}, id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0', usage_metadata={'input_tokens': 31, 'output_tokens': 5, 'total_tokens': 36})

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content="J'adore la programmation.", response_metadata={'finish_reason': 'stop'}, id='run-bf917526-7f58-4683-84f7-36a6b671d140', usage_metadata={'input_tokens': 31, 'output_tokens': 5, 'total_tokens': 36})

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(content="J'adore la programmation.", response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 31, 'total_tokens': 36}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_43dfabdef1', 'finish_reason': 'stop', 'logprobs': None}, id='run-012cffe2-5d3d-424d-83b5-51c6d4a593d1-0', usage_metadata={'input_tokens': 31, 'output_tokens': 5, 'total_tokens': 36})

    Tool calling:
        .. code-block:: python

            from langchain_core.pydantic_v1 import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetWeather',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'call_6XswGD5Pqk8Tt5atYr7tfenU'},
             {'name': 'GetWeather',
              'args': {'location': 'New York, NY'},
              'id': 'call_ZVL15vA8Y7kXqOy3dtmQgeCi'},
             {'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'call_49CFW8zqC9W7mh7hbMLSIrXw'},
             {'name': 'GetPopulation',
              'args': {'location': 'New York, NY'},
              'id': 'call_6ghfKxV264jEfe1mRIkS3PE7'}]

        See ``ChatOpenAI.bind_tools()`` method for more.

    Structured output:
        .. code-block:: python

            from typing import Optional

            from langchain_core.pydantic_v1 import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup='Why was the cat sitting on the computer?', punchline='To keep an eye on the mouse!', rating=None)

        See ``ChatOpenAI.with_structured_output()`` for more.

    JSON mode:
        .. code-block:: python

            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke("Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]")
            ai_msg.content

        .. code-block:: python

            '\\n{\\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\\n}'

    Image input:
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            "The weather in the image appears to be clear and pleasant. The sky is mostly blue with scattered, light clouds, suggesting a sunny day with minimal cloud cover. There is no indication of rain or strong winds, and the overall scene looks bright and calm. The lush green grass and clear visibility further indicate good weather conditions."

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 28, 'output_tokens': 5, 'total_tokens': 33}

    Logprobs:
        .. code-block:: python

            logprobs_llm = llm.bind(logprobs=True)
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

            {'content': [{'token': 'J', 'bytes': [74], 'logprob': -4.9617593e-06, 'top_logprobs': []},
              {'token': "'adore", 'bytes': [39, 97, 100, 111, 114, 101], 'logprob': -0.25202933, 'top_logprobs': []},
              {'token': ' la', 'bytes': [32, 108, 97], 'logprob': -0.20141791, 'top_logprobs': []},
              {'token': ' programmation', 'bytes': [32, 112, 114, 111, 103, 114, 97, 109, 109, 97, 116, 105, 111, 110], 'logprob': -1.9361265e-07, 'top_logprobs': []},
              {'token': '.', 'bytes': [46], 'logprob': -1.2233183e-05, 'top_logprobs': []}]}

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'token_usage': {'completion_tokens': 5,
              'prompt_tokens': 28,
              'total_tokens': 33},
             'model_name': 'gpt-4o',
             'system_fingerprint': 'fp_319be4768e',
             'finish_reason': 'stop',
             'logprobs': None}

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "openai"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.openai_api_base:
            attributes["openai_api_base"] = self.openai_api_base

        if self.openai_proxy:
            attributes["openai_proxy"] = self.openai_proxy

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
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
