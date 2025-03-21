from __future__ import annotations

import asyncio
import functools
import json
import logging
from operator import itemgetter
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
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
)
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
        tool_calls = []
        invalid_tool_calls = []
        if "tool_calls" in _dict:
            additional_kwargs = {"tool_calls": _dict["tool_calls"]}

            for index, value in enumerate(_dict["tool_calls"]):
                if is_chunk:
                    try:
                        tool_calls.append(
                            {
                                "name": value["function"].get("name"),
                                "args": value["function"].get("arguments"),
                                "id": value.get("id"),
                                # Tongyi does not respond with index,
                                # use index in the list instead
                                "index": index,
                            }
                        )
                    except KeyError:
                        pass
                else:
                    try:
                        parsed_tool = parse_tool_call(value, return_id=True)
                        if parsed_tool:
                            tool_calls.append(parsed_tool)
                    except Exception as e:
                        invalid_tool_calls.append(make_invalid_tool_call(value, str(e)))
        elif "reasoning_content" in _dict:
            additional_kwargs = {"reasoning_content": _dict["reasoning_content"]}
        elif "partial" in _dict and isinstance(_dict["partial"], bool):
            additional_kwargs = {"partial": _dict["partial"]}
        else:
            additional_kwargs = {}

        return (
            AIMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_call_chunks=tool_calls,  # type: ignore[arg-type]
                id=_dict.get("id"),
            )
            if is_chunk
            else AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_calls=tool_calls,  # type: ignore[arg-type]
                invalid_tool_calls=invalid_tool_calls,
            )
        )
    elif role == "system":
        return (
            SystemMessageChunk(content=content)
            if is_chunk
            else SystemMessage(content=content)
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return (
            ToolMessageChunk(
                content=_dict.get("content", ""),
                tool_call_id=_dict.get("tool_call_id"),  # type: ignore[arg-type]
                additional_kwargs=additional_kwargs,
            )
            if is_chunk
            else ToolMessage(
                content=_dict.get("content", ""),
                tool_call_id=_dict.get("tool_call_id"),  # type: ignore[arg-type]
                additional_kwargs=additional_kwargs,
            )
        )
    else:
        return (
            ChatMessageChunk(role=role, content=content)
            if is_chunk
            else ChatMessage(role=role, content=content)
        )


def convert_message_chunk_to_message(message_chunk: BaseMessageChunk) -> BaseMessage:
    """Convert a message chunk to a message.

    Args:
        chunk: Message chunk to convert.

    Returns:
        Message.
    """
    if not isinstance(message_chunk, BaseMessageChunk):
        return message_chunk
    # chunk classes always have the equivalent non-chunk class as their first parent
    ignore_keys = ["type"]
    if isinstance(message_chunk, AIMessageChunk):
        ignore_keys.append("tool_call_chunks")
    return message_chunk.__class__.__mro__[1](
        **{k: v for k, v in message_chunk.__dict__.items() if k not in ignore_keys}
    )


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
        # support Partial Mode for text continuation
        if "partial" in message.additional_kwargs:
            message_dict["partial"] = message.additional_kwargs["partial"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
            "name": message.name or message.additional_kwargs.get("name"),
        }
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "tool",
            "tool_call_id": "",
            "content": message.content,
            "name": message.name,
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


class ChatTongyi(BaseChatModel):
    """Alibaba Tongyi Qwen chat model integration.

    Setup:
        Install ``dashscope`` and set environment variables ``DASHSCOPE_API_KEY``.

        .. code-block:: bash

            pip install dashscope
            export DASHSCOPE_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Qianfan model to use.
        top_p: float
            Total probability mass of tokens to consider at each step.
        streaming: bool
            Whether to stream the results or not.

    Key init args — client params:
        api_key: Optional[str]
            Dashscope API KEY. If not passed in will be read from env var DASHSCOPE_API_KEY.
        max_retries: int
            Maximum number of retries to make when generating.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatTongyi

            tongyi_chat = ChatTongyi(
                model="qwen-max",
                # top_p="...",
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
                ("human", "我喜欢编程。"),
            ]
            tongyi_chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='I enjoy programming.',
                response_metadata={
                    'model_name': 'qwen-max',
                    'finish_reason': 'stop',
                    'request_id': '0bd14853-4abc-9593-8642-8dbb915bd4df',
                    'token_usage': {
                        'input_tokens': 30,
                        'output_tokens': 4,
                        'total_tokens': 34
                    }
                },
                id='run-533b3688-d12b-40c6-a2f7-52f291f8fa0a-0'
            )

    Stream:
        .. code-block:: python

            for chunk in tongyi_chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='I' id='run-8fbcce63-42fc-4208-9399-da46ac40c967'
            content=' enjoy' id='run-8fbcce63-42fc-4208-9399-da46ac40c967'
            content=' programming' id='run-8fbcce63-42fc-4208-9399-da46ac40c967'
            content='.' response_metadata={'finish_reason': 'stop', 'request_id': '67aec2b5-72bf-96a4-ae29-5bfebd2e7305', 'token_usage': {'input_tokens': 30, 'output_tokens': 4, 'total_tokens': 34}} id='run-8fbcce63-42fc-4208-9399-da46ac40c967'

    Async:
        .. code-block:: python

            await tongyi_chat.ainvoke(messages)

            # stream:
            # async for chunk in tongyi_chat.astream(messages):
            #    print(chunk)

            # batch:
            # await tongyi_chat.abatch([messages])

        .. code-block:: python

            AIMessage(
                content='I enjoy programming.',
                response_metadata={
                    'model_name': 'qwen-max',
                    'finish_reason': 'stop',
                    'request_id': 'a55a2d6c-a876-9789-9dd9-7b52bf8adde0',
                    'token_usage': {
                        'input_tokens': 30,
                        'output_tokens': 4,
                        'total_tokens': 34
                    }
                },
                id='run-3bffa3ec-e8d9-4043-b57d-348e047d64de-0'
            )

    Tool calling:
        .. code-block:: python

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

            chat_with_tools = tongyi_chat.bind_tools([GetWeather, GetPopulation])
            ai_msg = chat_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python
            [
                {
                    'name': 'GetWeather',
                    'args': {'location': 'Los Angeles, CA'},
                    'id': ''
                }
            ]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


            structured_chat = tongyi_chat.with_structured_output(Joke)
            structured_chat.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(
                setup='Why did the cat join the band?',
                punchline='Because it wanted to be a solo purr-sonality!',
                rating=None
            )

    Response metadata
        .. code-block:: python

            ai_msg = tongyi_chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'model_name': 'qwen-max',
                'finish_reason': 'stop',
                'request_id': '32a13e4c-370e-99cb-8f9b-4c999d98c57d',
                'token_usage': {
                    'input_tokens': 30,
                    'output_tokens': 4,
                    'total_tokens': 34
                }
            }

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    client: Any = None  #: :meta private:
    model_name: str = Field(default="qwen-turbo", alias="model")
    """Model name to use.
    callable multimodal model:
    - qwen-vl-v1
    - qwen-vl-chat-v1
    - qwen-audio-turbo
    - qwen-vl-plus
    - qwen-vl-max
    """
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[SecretStr] = Field(None, alias="api_key")
    """Dashscope api key provide by Alibaba Cloud."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @pre_init
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
        dashscope_multimodal_models = [
            "qwen-audio-turbo",
            "qwen-audio-turbo-latest",
            "qwen-vl-plus",
            "qwen-vl-plus-latest",
            "qwen-vl-max",
            "qwen-vl-max-latest",
        ]
        if (
            values["model_name"] in dashscope_multimodal_models
            or "vl" in values["model_name"]
        ):
            try:
                values["client"] = dashscope.MultiModalConversation
            except AttributeError:
                raise ValueError(
                    "`dashscope` has no `MultiModalConversation` attribute, this is "
                    "likely due to an old version of the dashscope package. Try "
                    "upgrading it with `pip install --upgrade dashscope`."
                )
        else:
            try:
                values["client"] = dashscope.Generation
            except AttributeError:
                raise ValueError(
                    "`dashscope` has no `Generation` attribute, this is likely "
                    "due to an old version of the dashscope package. Try upgrading it "
                    "with `pip install --upgrade dashscope`."
                )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Tongyi Qwen API."""
        return {
            "model": self.model_name,
            "top_p": self.top_p,
            "api_key": cast(SecretStr, self.dashscope_api_key).get_secret_value(),
            "result_format": "message",
            **self.model_kwargs,
        }

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
            prev_resp = None

            for resp in responses:
                # If we are streaming without `incremental_output = True`,
                # we need to calculate the delta response manually
                if _kwargs.get("stream") and not _kwargs.get(
                    "incremental_output", False
                ):
                    # inline fix response text logic
                    resp_copy = json.loads(json.dumps(resp))
                    if resp_copy.get("output") and resp_copy["output"].get("choices"):
                        choice = resp_copy["output"]["choices"][0]
                        message = choice["message"]
                        if isinstance(message.get("content"), list):
                            content_text = "".join(
                                item.get("text", "")
                                for item in message["content"]
                                if isinstance(item, dict)
                            )
                            message["content"] = content_text
                        resp = resp_copy
                    if prev_resp is None:
                        delta_resp = resp
                    else:
                        delta_resp = self.subtract_client_response(resp, prev_resp)
                    prev_resp = resp
                    yield check_response(delta_resp)
                else:
                    yield check_response(resp)

        return _stream_completion_with_retry(**kwargs)

    def subtract_client_response(self, resp: Any, prev_resp: Any) -> Any:
        """Subtract prev response from curr response.

        Useful when streaming without `incremental_output = True`
        """

        resp_copy = json.loads(json.dumps(resp))
        choice = resp_copy["output"]["choices"][0]
        message = choice["message"]

        prev_resp_copy = json.loads(json.dumps(prev_resp))
        prev_choice = prev_resp_copy["output"]["choices"][0]
        prev_message = prev_choice["message"]

        message["content"] = message["content"].replace(prev_message["content"], "")

        if message.get("tool_calls"):
            for index, tool_call in enumerate(message["tool_calls"]):
                function = tool_call["function"]

                if prev_message.get("tool_calls"):
                    prev_function = prev_message["tool_calls"][index]["function"]

                    function["name"] = function["name"].replace(
                        prev_function["name"], ""
                    )
                    function["arguments"] = function["arguments"].replace(
                        prev_function["arguments"], ""
                    )

        return resp_copy

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
            generation_chunk: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation_chunk is None:
                    generation_chunk = chunk
                else:
                    generation_chunk += chunk
            assert generation_chunk is not None
            generations.append(self._chunk_to_generation(generation_chunk))
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

        for stream_resp, is_last_chunk in generate_with_last_element_mark(
            self.stream_completion_with_retry(**params)
        ):
            choice = stream_resp["output"]["choices"][0]
            message = choice["message"]
            if (
                choice["finish_reason"] == "null"
                and message["content"] == ""
                and message["reasoning_content"] == ""
                and "tool_calls" not in message
            ):
                continue

            chunk = ChatGenerationChunk(
                **self._chat_generation_from_qwen_resp(
                    stream_resp, is_chunk=True, is_last_chunk=is_last_chunk
                )
            )
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
        async for stream_resp, is_last_chunk in agenerate_with_last_element_mark(
            self.astream_completion_with_retry(**params)
        ):
            chunk = ChatGenerationChunk(
                **self._chat_generation_from_qwen_resp(
                    stream_resp, is_chunk=True, is_last_chunk=is_last_chunk
                )
            )
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _invocation_params(
        self, messages: List[BaseMessage], stop: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        params = {**self._default_params, **kwargs}
        if stop is not None:
            params["stop"] = stop
        # According to the Tongyi official docs,
        # `incremental_output` with `tools` is not supported yet
        if params.get("stream") and not params.get("tools"):
            params["incremental_output"] = True

        message_dicts = [convert_message_to_dict(m) for m in messages]

        # And the `system` message should be the first message if present
        system_message_indices = [
            i for i, m in enumerate(message_dicts) if m["role"] == "system"
        ]
        if len(system_message_indices) == 1 and system_message_indices[0] != 0:
            raise ValueError("System message can only be the first message.")

        params["messages"] = message_dicts

        return params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        if llm_outputs[0] is None:
            return {}
        return llm_outputs[0]

    @staticmethod
    def _chat_generation_from_qwen_resp(
        resp: Any, is_chunk: bool = False, is_last_chunk: bool = True
    ) -> Dict[str, Any]:
        # According to the response from dashscope,
        # each chunk's `generation_info` overwrites the previous one.
        # Besides, The `merge_dicts` method,
        # which is used to concatenate `generation_info` in `GenerationChunk`,
        # does not support merging of int type values.
        # Therefore, we adopt the `generation_info` of the last chunk
        # and discard the `generation_info` of the intermediate chunks.
        choice = resp["output"]["choices"][0]
        message = convert_dict_to_message(choice["message"], is_chunk=is_chunk)
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
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec.
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

        """
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        llm = self.bind_tools([schema])
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema],  # type: ignore[list-item]
                first_tool_only=True,  # type: ignore[list-item]
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
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
