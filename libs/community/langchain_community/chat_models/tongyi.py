from __future__ import annotations

import asyncio
import functools
import logging
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
    Type,
    Union,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain_community.llms.tongyi import (
    agenerate_with_last_element_mark,
    check_response,
    generate_with_last_element_mark,
)

logger = logging.getLogger(__name__)


def convert_dict_to_message(
    _dict: Mapping[str, Any], is_chunk: bool = False
) -> Union[BaseMessage, BaseMessageChunk]:
    """Convert a dict to a message."""
    role = _dict["role"]
    content = _dict["content"]
    if role == "user":
        return (
            HumanMessageChunk(content=content)
            if is_chunk
            else HumanMessage(content=content)
        )
    elif role == "assistant":
        additional_kwargs = {}
        if is_chunk:
            tool_call_chunks = []
            if "tool_calls" in _dict:
                additional_kwargs["tool_calls"] = _dict["tool_calls"]
                for idx, raw_tool_call in enumerate(_dict["tool_calls"]):
                    tool_call_chunks.append(
                        {
                            "name": raw_tool_call.get("function", {}).get("name"),
                            "args": raw_tool_call.get("function", {}).get("arguments"),
                            "id": raw_tool_call.get("id"),
                            "index": idx,
                        }
                    )
            return _AITongyiMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_call_chunks=tool_call_chunks,
            )
        else:
            tool_calls = []
            invalid_tool_calls = []
            if "tool_calls" in _dict:
                additional_kwargs["tool_calls"] = _dict["tool_calls"]
                for raw_tool_call in _dict["tool_calls"]:
                    try:
                        tool_calls.append(
                            parse_tool_call(raw_tool_call, return_id=True)
                        )
                    except Exception as e:
                        invalid_tool_calls.append(
                            make_invalid_tool_call(raw_tool_call, str(e))
                        )
            return AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )
    elif role == "system":
        return (
            SystemMessageChunk(content=content)
            if is_chunk
            else SystemMessage(content=content)
        )
    else:
        return (
            ChatMessageChunk(role=role, content=content)
            if is_chunk
            else ChatMessage(role=role, content=content)
        )


def convert_message_chunk_to_message(message_chunk: BaseMessageChunk) -> BaseMessage:
    """Convert a message chunk to a message."""
    if isinstance(message_chunk, _AITongyiMessageChunk):
        return message_chunk_to_message(
            cast(AIMessageChunk, message_chunk_to_message(message_chunk))
        )
    else:
        return message_chunk_to_message(message_chunk)


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a message to a dict."""
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "name": message.name or message.additional_kwargs.get("name"),
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _create_retry_decorator(llm: ChatTongyi) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterward
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _remove_prefix(text: str, prefix: str) -> str:
    if prefix and text.startswith(prefix):
        return text[len(prefix) :]
    return text


class _AITongyiMessageChunk(AIMessageChunk):
    """Message chunk from Tongyi LLM,
    which handles the `tool_calls` stream appropriately.
    """

    type: Literal["_AITongyiMessageChunk"] = "_AITongyiMessageChunk"  # type: ignore[assignment] # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain_community", "chat_models"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        super_add_result = super().__add__(other)
        if isinstance(other, _AITongyiMessageChunk):
            return self.__class__(
                example=self.example,
                content=super_add_result.content,
                additional_kwargs=other.additional_kwargs,
                tool_call_chunks=other.tool_call_chunks,
                response_metadata=super_add_result.response_metadata,
                id=super_add_result.id,
            )
        return super_add_result


class ChatTongyi(BaseChatModel):
    """Alibaba Tongyi Qwen chat models API.

    To use, you should have the ``dashscope`` python package installed,
    and set env ``DASHSCOPE_API_KEY`` with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatTongyi
            Tongyi_chat = ChatTongyi()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    client: Any  #: :meta private:
    model_name: str = Field(default="qwen-turbo", alias="model")

    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[SecretStr] = Field(None, alias="api_key")
    """Dashscope api key provide by Alibaba Cloud."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["dashscope_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "dashscope_api_key", "DASHSCOPE_API_KEY")
        )
        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope --upgrade`."
            )
        try:
            values["client"] = dashscope.Generation
        except AttributeError:
            raise ValueError(
                "`dashscope` has no `Generation` attribute, this is likely "
                "due to an old version of the dashscope package. Try upgrading it "
                "with `pip install --upgrade dashscope`."
            )

        return values

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self)

        @retry_decorator
        def _completion_with_retry(**_kwargs: Any) -> Any:
            resp = self.client.call(**_kwargs)
            return check_response(resp)

        return _completion_with_retry(**kwargs)

    def stream_completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self)

        @retry_decorator
        def _stream_completion_with_retry(**_kwargs: Any) -> Any:
            responses = self.client.call(**_kwargs)
            for resp in responses:
                yield check_response(resp)

        return _stream_completion_with_retry(**kwargs)

    async def astream_completion_with_retry(self, **kwargs: Any) -> Any:
        """Because the dashscope SDK doesn't provide an async API,
        we wrap `stream_generate_with_retry` with an async generator."""

        class _AioTongyiGenerator:
            def __init__(self, generator: Any):
                self.generator = generator

            def __aiter__(self) -> AsyncIterator[Any]:
                return self

            async def __anext__(self) -> Any:
                value = await asyncio.get_running_loop().run_in_executor(
                    None, self._safe_next
                )
                if value is not None:
                    return value
                else:
                    raise StopAsyncIteration

            def _safe_next(self) -> Any:
                try:
                    return next(self.generator)
                except StopIteration:
                    return None

        async for chunk in _AioTongyiGenerator(
            generator=self.stream_completion_with_retry(**kwargs)
        ):
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        generations = []
        if self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            generations.append(self._chunk_to_generation(generation))
        else:
            params: Dict[str, Any] = self._invocation_params(
                messages=messages, stop=stop, **kwargs
            )
            resp = self.completion_with_retry(**params)
            generations.append(
                ChatGeneration(**self._chat_generation_from_qwen_resp(resp))
            )
        return ChatResult(
            generations=generations,
            llm_output={
                "model_name": self.model_name,
            },
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        generations = []
        if self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            async for chunk in self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            generations.append(self._chunk_to_generation(generation))
        else:
            params: Dict[str, Any] = self._invocation_params(
                messages=messages, stop=stop, **kwargs
            )
            resp = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(self.completion_with_retry, **params),
            )
            generations.append(
                ChatGeneration(**self._chat_generation_from_qwen_resp(resp))
            )
        return ChatResult(
            generations=generations,
            llm_output={
                "model_name": self.model_name,
            },
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(
            messages=messages, stop=stop, stream=True, **kwargs
        )
        incremental_output = params.get("incremental_output")
        previous_resp: Any = None
        for stream_resp, is_last_chunk in generate_with_last_element_mark(
            self.stream_completion_with_retry(**params)
        ):
            chunk = ChatGenerationChunk(
                **self._chat_generation_from_qwen_resp(
                    stream_resp,
                    previous_resp=previous_resp,
                    is_chunk=True,
                    is_last_chunk=is_last_chunk,
                )
            )
            if not incremental_output:
                previous_resp = stream_resp
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(
            messages=messages, stop=stop, stream=True, **kwargs
        )
        incremental_output = params.get("incremental_output")
        previous_resp: Any = None
        async for stream_resp, is_last_chunk in agenerate_with_last_element_mark(
            self.astream_completion_with_retry(**params)
        ):
            chunk = ChatGenerationChunk(
                **self._chat_generation_from_qwen_resp(
                    stream_resp,
                    previous_resp=previous_resp,
                    is_chunk=True,
                    is_last_chunk=is_last_chunk,
                )
            )
            if not incremental_output:
                previous_resp = stream_resp
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _invocation_params(
        self, messages: List[BaseMessage], stop: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        params = {
            "model": self.model_name,
            "top_p": self.top_p,
            "api_key": cast(SecretStr, self.dashscope_api_key).get_secret_value(),
            "result_format": "message",
            **self.model_kwargs,
            **kwargs,
        }
        if stop is not None:
            params["stop"] = stop

        # the default value of `incremental_output` is `False` in LLM API,
        # and it only works when `stream` is `True`.
        # So, to prevent some unexpected behavior,
        # we delete the `incremental_output` if it is unnecessary.
        if not params.get("stream") or not params.get("incremental_output"):
            if "incremental_output" in params:
                del params["incremental_output"]

        message_dicts = [convert_message_to_dict(m) for m in messages]

        # the `system` message should always be unique
        # and if present, it should be the first message
        system_message_indices = [
            i for i, m in enumerate(message_dicts) if m["role"] == "system"
        ]
        if len(system_message_indices) == 1 and system_message_indices[0] != 0:
            raise ValueError("System message can only be the first message.")
        elif len(system_message_indices) > 1:
            raise ValueError("There can be only one system message at most.")

        params["messages"] = message_dicts

        return params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return llm_outputs[0] or {}

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to bind.

        Example:
            .. code-block:: python

                from langchain_core.pydantic_v1 import BaseModel, Field
                from langchain_community.chat_models.tongyi import ChatTongyi

                class GetWeather(BaseModel):
                    '''Get the current weather in a given location'''

                    location: str = Field(
                        ...,
                        description="The city and state, e.g. San Francisco, CA",
                    )


                llm = ChatTongyi(model="qwen-max")
                llm_with_tools = llm.bind_tools([GetWeather])
                llm_with_tools.invoke("what is the weather like in HangZhou, China")

                # -> AIMessage(
                #    content='',
                #    id='run-f3bb9ff7-fbf5-43d4-880c-f28a5391c307-0',
                #    tool_calls=[{
                #        'name': 'GetWeather',
                #        'args': {'location': 'Hangzhou, China'},
                #        'id': ''
                #    }],
                #    response_metadata={
                #        'model_name': 'qwen-max',
                #        'finish_reason': 'tool_calls',
                #        ...
                #    }
                #    additional_kwargs={'tool_calls': [{...}]}
                # )
        """

        # According to the documentation of the dashscope:
        # 1. the `tools` parameter has exactly the same format
        #   as OpenAI, so we can use the `convert_to_openai_tool` function
        #   directly to convert the tools.
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # 2. the `incremental_output` parameter is not supported
        #   when `tools` are provided.
        return self.bind(tools=formatted_tools, incremental_output=False, **kwargs)

    @staticmethod
    def _chat_generation_from_qwen_resp(
        resp: Any,
        previous_resp: Any = None,
        is_chunk: bool = False,
        is_last_chunk: bool = True,
    ) -> Dict[str, Any]:
        choice = resp["output"]["choices"][0]
        raw_message = choice["message"]

        message_dict = {"role": raw_message["role"]}
        # if `previous_resp` is not None
        # (`incremental_output` should be False in this case),
        # we try to remove its content as the prefix of current response's content
        if previous_resp is not None:
            previous_content = previous_resp["output"]["choices"][0]["message"][
                "content"
            ]
            message_dict["content"] = _remove_prefix(
                raw_message["content"], prefix=previous_content
            )
        else:
            message_dict["content"] = raw_message["content"]
        if "tool_calls" in raw_message:
            message_dict["tool_calls"] = raw_message["tool_calls"]

        message = convert_dict_to_message(message_dict, is_chunk=is_chunk)

        # According to the response from dashscope,
        # each chunk's `generation_info` overwrites the previous one.
        # Besides, The `merge_dicts` method,
        # which is used to concatenate `generation_info` in `GenerationChunk`,
        # does not support merging of int type values.
        # Therefore, we adopt the `generation_info` of the last chunk
        # and discard the `generation_info` of the intermediate chunks.
        if is_last_chunk:
            return dict(
                message=message,
                generation_info=dict(
                    finish_reason=choice["finish_reason"],
                    request_id=resp["request_id"],
                    token_usage=dict(resp["usage"]),
                ),
            )
        else:
            return dict(message=message)

    @staticmethod
    def _chunk_to_generation(chunk: ChatGenerationChunk) -> ChatGeneration:
        return ChatGeneration(
            message=convert_message_chunk_to_message(chunk.message),
            generation_info=chunk.generation_info,
        )
