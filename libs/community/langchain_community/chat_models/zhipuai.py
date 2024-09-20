"""ZhipuAI chat models wrapper."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

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
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

API_TOKEN_TTL_SECONDS = 3 * 60
ZHIPUAI_API_BASE = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


@contextmanager
def connect_sse(client: Any, method: str, url: str, **kwargs: Any) -> Iterator:
    """Context manager for connecting to an SSE stream.

    Args:
        client: The HTTP client.
        method: The HTTP method.
        url: The URL.
        kwargs: Additional keyword arguments.

    Yields:
        The event source.
    """
    from httpx_sse import EventSource

    with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


@asynccontextmanager
async def aconnect_sse(
    client: Any, method: str, url: str, **kwargs: Any
) -> AsyncIterator:
    """Async context manager for connecting to an SSE stream.

    Args:
        client: The HTTP client.
        method: The HTTP method.
        url: The URL.
        kwargs: Additional keyword arguments.

    Yields:
        The event source.
    """
    from httpx_sse import EventSource

    async with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


def _get_jwt_token(api_key: str) -> str:
    """Gets JWT token for ZhipuAI API.

    See 'https://open.bigmodel.cn/dev/api#nosdk'.

    Args:
        api_key: The API key for ZhipuAI API.

    Returns:
        The JWT token.
    """
    try:
        import jwt
    except ImportError:
        raise ImportError(
            "jwt package not found, please install it with" "`pip install pyjwt`"
        )

    try:
        id, secret = api_key.split(".")
    except ValueError as err:
        raise ValueError(f"Invalid API key: {api_key}") from err

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + API_TOKEN_TTL_SECONDS * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


def _convert_dict_to_message(dct: Dict[str, Any]) -> BaseMessage:
    role = dct.get("role")
    content = dct.get("content", "")
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        additional_kwargs = {}
        tool_calls = dct.get("tool_calls", None)
        if tool_calls is not None:
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "tool":
        additional_kwargs = {}
        if "name" in dct:
            additional_kwargs["name"] = dct["name"]
        return ToolMessage(
            content=content,
            tool_call_id=dct.get("tool_call_id"),  # type: ignore[arg-type]
            additional_kwargs=additional_kwargs,
        )
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
            "name": message.name or message.additional_kwargs.get("name"),
        }
    else:
        raise TypeError(f"Got unknown type '{message.__class__.__name__}'.")
    return message_dict


def _convert_delta_to_message_chunk(
    dct: Dict[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = dct.get("role")
    content = dct.get("content", "")
    additional_kwargs = {}
    tool_calls = dct.get("tool_call", None)
    if tool_calls is not None:
        additional_kwargs["tool_calls"] = tool_calls

    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


def _truncate_params(payload: Dict[str, Any]) -> None:
    """Truncate temperature and top_p parameters between [0.01, 0.99].

    ZhipuAI only support temperature / top_p between (0, 1) open interval,
    so we truncate them to [0.01, 0.99].
    """
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    if temperature is not None:
        payload["temperature"] = max(0.01, min(0.99, temperature))
    if top_p is not None:
        payload["top_p"] = max(0.01, min(0.99, top_p))


class ChatZhipuAI(BaseChatModel):
    """ZhipuAI chat model integration.

    Setup:
        Install ``PyJWT`` and set environment variable ``ZHIPUAI_API_KEY``

        .. code-block:: bash

            pip install pyjwt
            export ZHIPUAI_API_KEY="your-api-key"

    Key init args — completion params:
        model: Optional[str]
            Name of ZhipuAI model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        api_key: Optional[str]
            ZhipuAI API key. If not passed in will be read from env var ZHIPUAI_API_KEY.
        api_base: Optional[str]
            Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatZhipuAI

            zhipuai_chat = ChatZhipuAI(
                temperature=0.5,
                api_key="your-api-key",
                model="glm-4",
                # api_base="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
                ("human", "我喜欢编程。"),
            ]
            zhipuai_chat.invoke(messages)

        .. code-block:: python

            AIMessage(content='I enjoy programming.', response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 23, 'total_tokens': 29}, 'model_name': 'glm-4', 'finish_reason': 'stop'}, id='run-c5d9af91-55c6-470e-9545-02b2fa0d7f9d-0')

    Stream:
        .. code-block:: python

            for chunk in zhipuai_chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='I' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content=' enjoy' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content=' programming' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content='.' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content='' response_metadata={'finish_reason': 'stop'} id='run-4df71729-618f-4e2b-a4ff-884682723082'

        .. code-block:: python

            stream = zhipuai_chat.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block::

            AIMessageChunk(content='I enjoy programming.', response_metadata={'finish_reason': 'stop'}, id='run-20b05040-a0b4-4715-8fdc-b39dba9bfb53')

    Async:
        .. code-block:: python

            await zhipuai_chat.ainvoke(messages)

            # stream:
            # async for chunk in zhipuai_chat.astream(messages):
            #    print(chunk)

            # batch:
            # await zhipuai_chat.abatch([messages])

        .. code-block:: python

            [AIMessage(content='I enjoy programming.', response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 23, 'total_tokens': 29}, 'model_name': 'glm-4', 'finish_reason': 'stop'}, id='run-ba06af9d-4baa-40b2-9298-be9c62aa0849-0')]

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

            chat_with_tools = zhipuai_chat.bind_tools([GetWeather, GetPopulation])
            ai_msg = chat_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python

            [
                {
                    'name': 'GetWeather',
                    'args': {'location': 'Los Angeles, CA'},
                    'id': 'call_202408222146464ea49ec8731145a9',
                    'type': 'tool_call'
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


            structured_chat = zhipuai_chat.with_structured_output(Joke)
            structured_chat.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup='What do cats like to eat for breakfast?', punchline='Mice Krispies!', rating=None)

    Response metadata
        .. code-block:: python

            ai_msg = zhipuai_chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'token_usage': {'completion_tokens': 6,
              'prompt_tokens': 23,
              'total_tokens': 29},
              'model_name': 'glm-4',
              'finish_reason': 'stop'}

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"zhipuai_api_key": "ZHIPUAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "zhipuai"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.zhipuai_api_base:
            attributes["zhipuai_api_base"] = self.zhipuai_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "zhipuai-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    # client:
    zhipuai_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `ZHIPUAI_API_KEY` if not provided."""
    zhipuai_api_base: Optional[str] = Field(default=None, alias="api_base")
    """Base URL path for API requests, leave blank if not using a proxy or service
        emulator.
    """

    model_name: Optional[str] = Field(default="glm-4", alias="model")
    """
    Model name to use, see 'https://open.bigmodel.cn/dev/api#language'.
    Alternatively, you can use any fine-tuned model from the GLM series.
    """

    temperature: float = 0.95
    """
    What sampling temperature to use. The value ranges from 0.0 to 1.0 and cannot
    be equal to 0.
    The larger the value, the more random and creative the output; The smaller
    the value, the more stable or certain the output will be.
    You are advised to adjust top_p or temperature parameters based on application
    scenarios, but do not adjust the two parameters at the same time.
    """

    top_p: float = 0.7
    """
    Another method of sampling temperature is called nuclear sampling. The value
    ranges from 0.0 to 1.0 and cannot be equal to 0 or 1.
    The model considers the results with top_p probability quality tokens.
    For example, 0.1 means that the model decoder only considers tokens from the
    top 10% probability of the candidate set.
    You are advised to adjust top_p or temperature parameters based on application
    scenarios, but do not adjust the two parameters at the same time.
    """

    streaming: bool = False
    """Whether to stream the results or not."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Any:
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values, ["zhipuai_api_key", "api_key"], "ZHIPUAI_API_KEY"
        )
        values["zhipuai_api_base"] = get_from_dict_or_env(
            values, "zhipuai_api_base", "ZHIPUAI_API_BASE", default=ZHIPUAI_API_BASE
        )

        return values

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        if self.zhipuai_api_key is None:
            raise ValueError("Did not find zhipuai_api_key.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {
            **params,
            **kwargs,
            "messages": message_dicts,
            "stream": False,
        }
        _truncate_params(payload)
        headers = {
            "Authorization": _get_jwt_token(self.zhipuai_api_key),
            "Accept": "application/json",
        }
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            response = client.post(self.zhipuai_api_base, json=payload)  # type: ignore[arg-type]
            response.raise_for_status()
        return self._create_chat_result(response.json())

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response in chunks."""
        if self.zhipuai_api_key is None:
            raise ValueError("Did not find zhipuai_api_key.")
        if self.zhipuai_api_base is None:
            raise ValueError("Did not find zhipu_api_base.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": True}
        _truncate_params(payload)
        headers = {
            "Authorization": _get_jwt_token(self.zhipuai_api_key),
            "Accept": "application/json",
        }

        default_chunk_class = AIMessageChunk
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            with connect_sse(
                client, "POST", self.zhipuai_api_base, json=payload
            ) as event_source:
                for sse in event_source.iter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], default_chunk_class
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
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

                    if finish_reason is not None:
                        break

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

        if self.zhipuai_api_key is None:
            raise ValueError("Did not find zhipuai_api_key.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {
            **params,
            **kwargs,
            "messages": message_dicts,
            "stream": False,
        }
        _truncate_params(payload)
        headers = {
            "Authorization": _get_jwt_token(self.zhipuai_api_key),
            "Accept": "application/json",
        }
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            response = await client.post(self.zhipuai_api_base, json=payload)  # type: ignore[arg-type]
            response.raise_for_status()
        return self._create_chat_result(response.json())

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.zhipuai_api_key is None:
            raise ValueError("Did not find zhipuai_api_key.")
        if self.zhipuai_api_base is None:
            raise ValueError("Did not find zhipu_api_base.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": True}
        _truncate_params(payload)
        headers = {
            "Authorization": _get_jwt_token(self.zhipuai_api_key),
            "Accept": "application/json",
        }

        default_chunk_class = AIMessageChunk
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async with aconnect_sse(
                client, "POST", self.zhipuai_api_base, json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], default_chunk_class
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

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.
        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Currently this can only be auto for this chat model.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        if self.model_name == "glm-4v":
            raise ValueError("glm-4v currently does not support tool calling")

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice and tool_choice != "auto":
            raise ValueError("ChatZhipuAI currently only supports `auto` tool choice")
        elif tool_choice and tool_choice == "auto":
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
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
            method: The method for steering model generation, either "function_calling"
                or "json_mode". ZhipuAI only supports "function_calling" which
                converts the schema to a OpenAI function and the model will make use of the
                function-calling API.
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

                from langchain_community.chat_models import ChatZhipuAI
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatZhipuAI(temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> AnswerWithJustification(
                #     answer='A pound of bricks and a pound of feathers weigh the same.'
                #     justification="Both a pound of bricks and a pound of feathers have been defined to have the same weight. The 'pound' is a unit of weight, so any two things that are described as weighing a pound will weigh the same."
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_community.chat_models import ChatZhipuAI
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatZhipuAI(temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_01htjn3cspevxbqc1d7nkk8wab', 'function': {'arguments': '{"answer": "A pound of bricks and a pound of feathers weigh the same.", "justification": "Both a pound of bricks and a pound of feathers have been defined to have the same weight. The \'pound\' is a unit of weight, so any two things that are described as weighing a pound will weigh the same.", "unit": "pounds"}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}, id='run-456beee6-65f6-4e80-88af-a6065480822c-0'),
                #     'parsed': AnswerWithJustification(answer='A pound of bricks and a pound of feathers weigh the same.', justification="Both a pound of bricks and a pound of feathers have been defined to have the same weight. The 'pound' is a unit of weight, so any two things that are described as weighing a pound will weigh the same."),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_community.chat_models import ChatZhipuAI
                from pydantic import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatZhipuAI(temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'A pound of bricks and a pound of feathers weigh the same.',
                #     'justification': "Both a pound of bricks and a pound of feathers have been defined to have the same weight. The 'pound' is a unit of weight, so any two things that are described as weighing a pound will weigh the same.", 'unit': 'pounds'}
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
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools([schema], tool_choice="auto")
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        else:
            raise ValueError(
                f"""Unrecognized method argument. Expected 'function_calling'.
                Received: '{method}'"""
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
