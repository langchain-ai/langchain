"""ZhipuAI chat models wrapper."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

API_TOKEN_TTL_SECONDS = 3 * 60
ZHIPUAI_API_BASE = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


@contextmanager
def connect_sse(client: Any, method: str, url: str, **kwargs: Any) -> Iterator:
    """Context manager for connecting to an SSE stream.

    Args:
        client: The HTTP client.
        method: The HTTP method.
        url: The URL.
        **kwargs: Additional keyword arguments.

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
        **kwargs: Additional keyword arguments.

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
    import jwt

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
            Name of OpenAI model to use.
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

            stream = llm.stream(messages)
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

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
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
            response = client.post(self.zhipuai_api_base, json=payload)
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
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
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
            response = await client.post(self.zhipuai_api_base, json=payload)
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
                    yield chunk
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    if finish_reason is not None:
                        break
