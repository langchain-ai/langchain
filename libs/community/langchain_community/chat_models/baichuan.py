import json
import logging
from contextlib import asynccontextmanager
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
)

import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
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
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_community.chat_models.llamacpp import (
    _lc_invalid_tool_call_to_openai_tool_call,
    _lc_tool_call_to_openai_tool_call,
)

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.baichuan-ai.com/v1/chat/completions"


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    content = message.content
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": content}
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]

        elif message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": content,
            "name": message.name or message.additional_kwargs.get("name"),
        }

    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    content = _dict.get("content", "")
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        tool_calls = []
        invalid_tool_calls = []
        additional_kwargs = {}

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
            tool_calls=tool_calls,  # type: ignore[arg-type]
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=content,
            tool_call_id=_dict.get("tool_call_id"),  # type: ignore[arg-type]
            additional_kwargs=additional_kwargs,
        )
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return ChatMessage(content=content, role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


@asynccontextmanager
async def aconnect_httpx_sse(
    client: Any, method: str, url: str, **kwargs: Any
) -> AsyncIterator:
    """Async context manager for connecting to an SSE stream.

    Args:
        client: The httpx client.
        method: The HTTP method.
        url: The URL to connect to.
        kwargs: Additional keyword arguments to pass to the client.

    Yields:
        An EventSource object.
    """
    from httpx_sse import EventSource

    async with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


class ChatBaichuan(BaseChatModel):
    """Baichuan chat model integration.

    Setup:
        To use, you should have the environment variable``BAICHUAN_API_KEY`` set with
    your API KEY.

        .. code-block:: bash

            export BAICHUAN_API_KEY="your-api-key"

    Key init args — completion params:
        model: Optional[str]
            Name of Baichuan model to use.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        streaming: Optional[bool]
            Whether to stream the results or not.
        temperature: Optional[float]
            Sampling temperature.
        top_p: Optional[float]
            What probability mass to use.
        top_k: Optional[int]
            What search sampling control to use.

    Key init args — client params:
        api_key: Optional[str]
            Baichuan API key. If not passed in will be read from env var BAICHUAN_API_KEY.
        base_url: Optional[str]
            Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatBaichuan

            chat = ChatBaichuan(
                api_key=api_key,
                model='Baichuan4',
                # temperature=...,
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
                ("human", "我喜欢编程。"),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='I enjoy programming.',
                response_metadata={
                    'token_usage': {
                        'prompt_tokens': 93,
                        'completion_tokens': 5,
                        'total_tokens': 98
                    },
                    'model': 'Baichuan4'
                },
                id='run-944ff552-6a93-44cf-a861-4e4d849746f9-0'
            )

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='I' id='run-f99fcd6f-dd31-46d5-be8f-0b6a22bf77d8'
            content=' enjoy programming.' id='run-f99fcd6f-dd31-46d5-be8f-0b6a22bf77d8

        .. code-block:: python

            stream = chat.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content='I like programming.',
                id='run-74689970-dc31-461d-b729-3b6aa93508d2'
            )

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

            # stream
            # async for chunk in chat.astream(messages):
            #     print(chunk)

            # batch
            # await chat.abatch([messages])

        .. code-block:: python

            AIMessage(
                content='I enjoy programming.',
                response_metadata={
                    'token_usage': {
                        'prompt_tokens': 93,
                        'completion_tokens': 5,
                        'total_tokens': 98
                    },
                    'model': 'Baichuan4'
                },
                id='run-952509ed-9154-4ff9-b187-e616d7ddfbba-0'
            )
    Tool calling:

        .. code-block:: python
            class get_current_weather(BaseModel):
                '''Get current weather.'''

                location: str = Field('City or province, such as Shanghai')


            llm_with_tools = ChatBaichuan(model='Baichuan3-Turbo').bind_tools([get_current_weather])
            llm_with_tools.invoke('How is the weather today?')

        .. code-block:: python

            [{'name': 'get_current_weather',
            'args': {'location': 'New York'},
            'id': '3951017OF8doB0A',
            'type': 'tool_call'}]

    Response metadata
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'token_usage': {
                    'prompt_tokens': 93,
                    'completion_tokens': 5,
                    'total_tokens': 98
                },
                'model': 'Baichuan4'
            }

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "baichuan_api_key": "BAICHUAN_API_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    baichuan_api_base: str = Field(default=DEFAULT_API_BASE, alias="base_url")
    """Baichuan custom endpoints"""
    baichuan_api_key: SecretStr = Field(alias="api_key")
    """Baichuan API Key"""
    baichuan_secret_key: Optional[SecretStr] = None
    """[DEPRECATED, keeping it for for backward compatibility] Baichuan Secret Key"""
    streaming: bool = False
    """Whether to stream the results or not."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    request_timeout: int = Field(default=60, alias="timeout")
    """request timeout for chat http requests"""
    model: str = "Baichuan2-Turbo-192K"
    """model name of Baichuan, default is `Baichuan2-Turbo-192K`,
    other options include `Baichuan2-Turbo`"""
    temperature: Optional[float] = Field(default=0.3)
    """What sampling temperature to use."""
    top_k: int = 5
    """What search sampling control to use."""
    top_p: float = 0.85
    """What probability mass to use."""
    with_search_enhance: bool = False
    """[DEPRECATED, keeping it for for backward compatibility], 
    Whether to use search enhance, default is False."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    class Config:
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        values["baichuan_api_base"] = get_from_dict_or_env(
            values,
            "baichuan_api_base",
            "BAICHUAN_API_BASE",
            DEFAULT_API_BASE,
        )
        values["baichuan_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                ["baichuan_api_key", "api_key"],
                "BAICHUAN_API_KEY",
            )
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Baichuan API."""
        normal_params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": self.streaming,
            "max_tokens": self.max_tokens,
        }

        return {**normal_params, **self.model_kwargs}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        res = self._chat(messages, **kwargs)
        if res.status_code != 200:
            raise ValueError(f"Error from Baichuan api response: {res}")
        response = res.json()
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        res = self._chat(messages, stream=True, **kwargs)
        if res.status_code != 200:
            raise ValueError(f"Error from Baichuan api response: {res}")
        default_chunk_class = AIMessageChunk
        for chunk in res.iter_lines():
            chunk = chunk.decode("utf-8").strip("\r\n")
            parts = chunk.split("data: ", 1)
            chunk = parts[1] if len(parts) > 1 else None
            if chunk is None:
                continue
            if chunk == "[DONE]":
                break
            response = json.loads(chunk)
            for m in response.get("choices"):
                chunk = _convert_delta_to_message_chunk(
                    m.get("delta"), default_chunk_class
                )
                default_chunk_class = chunk.__class__
                cg_chunk = ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
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

        headers = self._create_headers_parameters(**kwargs)
        payload = self._create_payload_parameters(messages, **kwargs)

        import httpx

        async with httpx.AsyncClient(
            headers=headers, timeout=self.request_timeout
        ) as client:
            response = await client.post(self.baichuan_api_base, json=payload)
            response.raise_for_status()
        return self._create_chat_result(response.json())

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        headers = self._create_headers_parameters(**kwargs)
        payload = self._create_payload_parameters(messages, stream=True, **kwargs)
        import httpx

        async with httpx.AsyncClient(
            headers=headers, timeout=self.request_timeout
        ) as client:
            async with aconnect_httpx_sse(
                client, "POST", self.baichuan_api_base, json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], AIMessageChunk
                    )
                    finish_reason = choice.get("finish_reason", None)

                    generation_info = (
                        {"finish_reason": finish_reason}
                        if finish_reason is not None
                        else None
                    )
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info
                    )
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk
                    if finish_reason is not None:
                        break

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        payload = self._create_payload_parameters(messages, **kwargs)
        url = self.baichuan_api_base
        headers = self._create_headers_parameters(**kwargs)

        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers=headers,
            json=payload,
            stream=self.streaming,
        )
        return res

    def _create_payload_parameters(  # type: ignore[no-untyped-def]
        self, messages: List[BaseMessage], **kwargs
    ) -> Dict[str, Any]:
        parameters = {**self._default_params, **kwargs}
        temperature = parameters.pop("temperature", 0.3)
        top_k = parameters.pop("top_k", 5)
        top_p = parameters.pop("top_p", 0.85)
        model = parameters.pop("model")
        with_search_enhance = parameters.pop("with_search_enhance", False)
        stream = parameters.pop("stream", False)
        tools = parameters.pop("tools", [])

        payload = {
            "model": model,
            "messages": [_convert_message_to_dict(m) for m in messages],
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "with_search_enhance": with_search_enhance,
            "stream": stream,
            "tools": tools,
        }

        return payload

    def _create_headers_parameters(self, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        parameters = {**self._default_params, **kwargs}
        default_headers = parameters.pop("headers", {})
        api_key = ""
        if self.baichuan_api_key:
            api_key = self.baichuan_api_key.get_secret_value()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **default_headers,
        }
        return headers

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for c in response["choices"]:
            message = _convert_dict_to_message(c["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)

        token_usage = response["usage"]
        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "baichuan-chat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, callable, or BaseTool.
                Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
