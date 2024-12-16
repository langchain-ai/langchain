import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Type

import requests
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
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from pydantic import ConfigDict, Field, SecretStr

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_CN = "https://api.lingyiwanwu.com/v1/chat/completions"
DEFAULT_API_BASE_GLOBAL = "https://api.01.ai/v1/chat/completions"


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    elif role == "system":
        return AIMessage(content=_dict["content"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role: str = _dict["role"]
    content = _dict.get("content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content, type=role)


@asynccontextmanager
async def aconnect_httpx_sse(
    client: Any, method: str, url: str, **kwargs: Any
) -> AsyncIterator:
    from httpx_sse import EventSource

    async with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


class ChatYi(BaseChatModel):
    """Yi chat models API."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "yi_api_key": "YI_API_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    yi_api_base: str = Field(default=DEFAULT_API_BASE_CN)
    yi_api_key: SecretStr = Field(alias="api_key")
    region: str = Field(default="cn")  # 默认使用中国区
    streaming: bool = False
    request_timeout: int = Field(default=60, alias="timeout")
    model: str = "yi-large"
    temperature: Optional[float] = Field(default=0.7)
    top_p: float = 0.7
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        populate_by_name=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs["yi_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                kwargs,
                ["yi_api_key", "api_key"],
                "YI_API_KEY",
            )
        )
        if kwargs.get("yi_api_base") is None:
            region = kwargs.get("region", "cn").lower()
            if region == "global":
                kwargs["yi_api_base"] = DEFAULT_API_BASE_GLOBAL
            else:
                kwargs["yi_api_base"] = DEFAULT_API_BASE_CN

        all_required_field_names = get_pydantic_field_names(self.__class__)
        extra = kwargs.get("model_kwargs", {})
        for field_name in list(kwargs):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                extra[field_name] = kwargs.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        kwargs["model_kwargs"] = extra
        super().__init__(**kwargs)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.streaming,
        }

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
            raise ValueError(f"Error from Yi api response: {res}")
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
            raise ValueError(f"Error from Yi api response: {res}")
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
            response = await client.post(self.yi_api_base, json=payload)
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
                client, "POST", self.yi_api_base, json=payload
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
        url = self.yi_api_base
        headers = self._create_headers_parameters(**kwargs)

        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers=headers,
            json=payload,
            stream=self.streaming,
        )
        return res

    def _create_payload_parameters(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        parameters = {**self._default_params, **kwargs}
        temperature = parameters.pop("temperature", 0.7)
        top_p = parameters.pop("top_p", 0.7)
        model = parameters.pop("model")
        stream = parameters.pop("stream", False)

        payload = {
            "model": model,
            "messages": [_convert_message_to_dict(m) for m in messages],
            "top_p": top_p,
            "temperature": temperature,
            "stream": stream,
        }
        return payload

    def _create_headers_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        parameters = {**self._default_params, **kwargs}
        default_headers = parameters.pop("headers", {})
        api_key = ""
        if self.yi_api_key:
            api_key = self.yi_api_key.get_secret_value()

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
        return "yi-chat"
