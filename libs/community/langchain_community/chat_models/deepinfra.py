"""deepinfra.com chat models wrapper"""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import aiohttp
import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
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
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_community.utilities.requests import Requests

logger = logging.getLogger(__name__)


class ChatDeepInfraException(Exception):
    """Exception raised when the DeepInfra API returns an error."""

    pass


def _create_retry_decorator(
    llm: ChatDeepInfra,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle PaLM exceptions."""
    return create_base_retry_decorator(
        error_types=[requests.exceptions.ConnectTimeout, ChatDeepInfraException],
        max_retries=llm.max_retries,
        run_manager=run_manager,
    )


def _parse_tool_calling(tool_call: dict) -> ToolCall:
    """
    Convert a tool calling response from server to a ToolCall object.
    Args:
        tool_call:

    Returns:

    """
    name = tool_call["function"].get("name", "")
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except (JSONDecodeError, TypeError):
        args = {}
    id = tool_call.get("id")
    return create_tool_call(name=name, args=args, id=id)


def _convert_to_tool_calling(tool_call: ToolCall) -> Dict[str, Any]:
    """
    Convert a ToolCall object to a tool calling request for server.
    Args:
        tool_call:

    Returns:

    """
    return {
        "type": "function",
        "function": {
            "arguments": json.dumps(tool_call["args"]),
            "name": tool_call["name"],
        },
        "id": tool_call.get("id"),
    }


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict.get("content", "") or ""
        tool_calls_content = _dict.get("tool_calls", []) or []
        tool_calls = [
            _parse_tool_calling(tool_call) for tool_call in tool_calls_content
        ]
        return AIMessage(content=content, tool_calls=tool_calls)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    tool_calls = _dict.get("tool_calls") or []

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        tool_calls = [_parse_tool_calling(tool_call) for tool_call in tool_calls]
        return AIMessageChunk(content=content, tool_calls=tool_calls)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        tool_calls = [
            _convert_to_tool_calling(tool_call) for tool_call in message.tool_calls
        ]
        message_dict = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls,  # type: ignore[dict-item]
        }
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "name": message.name,  # type: ignore[dict-item]
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatDeepInfra(BaseChatModel):
    """A chat model that uses the DeepInfra API."""

    # client: Any  #: :meta private:
    model_name: str = Field(default="meta-llama/Llama-2-70b-chat-hf", alias="model")
    """Model name to use."""

    url: str = "https://api.deepinfra.com/v1/openai/chat/completions"
    """URL to use for the API call."""

    deepinfra_api_token: Optional[str] = None
    request_timeout: Optional[float] = Field(default=None, alias="timeout")
    temperature: Optional[float] = 1
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Run inference with this temperature. Must be in the closed
       interval [0.0, 1.0]."""
    top_p: Optional[float] = None
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: Optional[int] = None
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    n: int = 1
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""
    max_tokens: int = 256
    streaming: bool = False
    max_retries: int = 1

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            **self.model_kwargs,
        }

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the openai client."""
        return {**self._default_params}

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            try:
                request_timeout = kwargs.pop("request_timeout")
                request = Requests(headers=self._headers())
                response = request.post(
                    url=self._url(), data=self._body(kwargs), timeout=request_timeout
                )
                self._handle_status(response.status_code, response.text)
                return response
            except Exception as e:
                print("EX", e)  # noqa: T201
                raise

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            try:
                request_timeout = kwargs.pop("request_timeout")
                request = Requests(headers=self._headers())
                async with request.apost(
                    url=self._url(), data=self._body(kwargs), timeout=request_timeout
                ) as response:
                    self._handle_status(response.status, await response.text())
                    return await response.json()
            except Exception as e:
                print("EX", e)  # noqa: T201
                raise

        return await _completion_with_retry(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def init_defaults(cls, values: Dict) -> Any:
        """Validate api key, python package exists, temperature, top_p, and top_k."""
        # For compatibility with LiteLLM
        api_key = get_from_dict_or_env(
            values,
            "deepinfra_api_key",
            "DEEPINFRA_API_KEY",
            default="",
        )
        values["deepinfra_api_token"] = get_from_dict_or_env(
            values,
            "deepinfra_api_token",
            "DEEPINFRA_API_TOKEN",
            default=api_key,
        )
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response.json())

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model": self.model_name}
        res = ChatResult(generations=generations, llm_output=llm_output)
        return res

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        for line in _parse_stream(response.iter_lines()):
            chunk = _handle_sse_line(line)
            if chunk:
                cg_chunk = ChatGenerationChunk(message=chunk, generation_info=None)
                if run_manager:
                    run_manager.on_llm_new_token(str(chunk.content), chunk=cg_chunk)
                yield cg_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {"messages": message_dicts, "stream": True, **params, **kwargs}

        request_timeout = params.pop("request_timeout")
        request = Requests(headers=self._headers())
        async with request.apost(
            url=self._url(), data=self._body(params), timeout=request_timeout
        ) as response:
            async for line in _parse_stream_async(response.content):
                chunk = _handle_sse_line(line)
                if chunk:
                    cg_chunk = ChatGenerationChunk(message=chunk, generation_info=None)
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            str(chunk.content), chunk=cg_chunk
                        )
                    yield cg_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {"messages": message_dicts, **params, **kwargs}

        res = await self.acompletion_with_retry(run_manager=run_manager, **params)
        return self._create_chat_result(res)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n": self.n,
        }

    @property
    def _llm_type(self) -> str:
        return "deepinfra-chat"

    def _handle_status(self, code: int, text: Any) -> None:
        if code >= 500:
            raise ChatDeepInfraException(
                f"DeepInfra Server error status {code}: {text}"
            )
        elif code >= 400:
            raise ValueError(f"DeepInfra received an invalid payload: {text}")
        elif code != 200:
            raise Exception(
                f"DeepInfra returned an unexpected response with status {code}: {text}"
            )

    def _url(self) -> str:
        return self.url

    def _headers(self) -> Dict:
        return {
            "Authorization": f"bearer {self.deepinfra_api_token}",
            "Content-Type": "application/json",
        }

    def _body(self, kwargs: Any) -> Dict:
        return kwargs

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)


def _parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = _parse_stream_helper(line)
        if _line is not None:
            yield _line


async def _parse_stream_async(rbody: aiohttp.StreamReader) -> AsyncIterator[str]:
    async for line in rbody:
        _line = _parse_stream_helper(line)
        if _line is not None:
            yield _line


def _parse_stream_helper(line: bytes) -> Optional[str]:
    if line and line.startswith(b"data:"):
        if line.startswith(b"data: "):
            # SSE event may be valid when it contain whitespace
            line = line[len(b"data: ") :]
        else:
            line = line[len(b"data:") :]
        if line.strip() == b"[DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        else:
            return line.decode("utf-8")
    return None


def _handle_sse_line(line: str) -> Optional[BaseMessageChunk]:
    try:
        obj = json.loads(line)
        default_chunk_class = AIMessageChunk
        delta = obj.get("choices", [{}])[0].get("delta", {})
        return _convert_delta_to_message_chunk(delta, default_chunk_class)
    except Exception:
        return None
