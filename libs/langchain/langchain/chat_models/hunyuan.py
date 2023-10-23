import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type, Union
from urllib.parse import urlparse

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel, _generate_from_stream
from langchain.pydantic_v1 import Field, SecretStr, root_validator
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
)
from langchain.schema.messages import (
    AIMessageChunk,
    BaseMessageChunk,
    ChatMessageChunk,
    HumanMessageChunk,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.utils import get_from_dict_or_env, get_pydantic_field_names

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://hunyuan.cloud.tencent.com"
DEFAULT_PATH = "/hyllm/v1/chat/completions"


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
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
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


# signature generation
# https://cloud.tencent.com/document/product/1729/97732#532252ce-e960-48a7-8821-940a9ce2ccf3
def _signature(secret_key: SecretStr, url: str, payload: Dict[str, Any]) -> str:
    sorted_keys = sorted(payload.keys())

    url_info = urlparse(url)

    sign_str = url_info.netloc + url_info.path + "?"

    for key in sorted_keys:
        value = payload[key]

        if isinstance(value, list) or isinstance(value, dict):
            value = json.dumps(value, separators=(",", ":"))
        elif isinstance(value, float):
            value = "%g" % value

        sign_str = sign_str + key + "=" + str(value) + "&"

    sign_str = sign_str[:-1]

    hmacstr = hmac.new(
        key=secret_key.get_secret_value().encode("utf-8"),
        msg=sign_str.encode("utf-8"),
        digestmod=hashlib.sha1,
    ).digest()

    return base64.b64encode(hmacstr).decode("utf-8")


def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
    generations = []
    for choice in response["choices"]:
        message = _convert_dict_to_message(choice["messages"])
        generations.append(ChatGeneration(message=message))

    token_usage = response["usage"]
    llm_output = {"token_usage": token_usage}
    return ChatResult(generations=generations, llm_output=llm_output)


def _to_secret(value: Union[SecretStr, str]) -> SecretStr:
    """Convert a string to a SecretStr if needed."""
    if isinstance(value, SecretStr):
        return value
    return SecretStr(value)


class ChatHunyuan(BaseChatModel):
    """Tencent Hunyuan chat models API by Tencent.

    For more information, see https://cloud.tencent.com/document/product/1729
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "hunyuan_app_id": "HUNYUAN_APP_ID",
            "hunyuan_secret_id": "HUNYUAN_SECRET_ID",
            "hunyuan_secret_key": "HUNYUAN_SECRET_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    hunyuan_api_base: str = Field(default=DEFAULT_API_BASE)
    """Hunyuan custom endpoints"""
    hunyuan_app_id: Optional[str] = None
    """Hunyuan App ID"""
    hunyuan_secret_id: Optional[str] = None
    """Hunyuan Secret ID"""
    hunyuan_secret_key: Optional[SecretStr] = None
    """Hunyuan Secret Key"""
    streaming: bool = False
    """Whether to stream the results or not."""
    request_timeout: int = 60
    """Timeout for requests to Hunyuan API. Default is 60 seconds."""

    query_id: Optional[str] = None
    """Query id for troubleshooting"""
    temperature: float = 1.0
    """What sampling temperature to use."""
    top_p: float = 1.0
    """What probability mass to use."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    class Config:
        """Configuration for this pydantic object."""

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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["hunyuan_api_base"] = get_from_dict_or_env(
            values,
            "hunyuan_api_base",
            "HUNYUAN_API_BASE",
            DEFAULT_API_BASE,
        )
        values["hunyuan_app_id"] = get_from_dict_or_env(
            values,
            "hunyuan_app_id",
            "HUNYUAN_APP_ID",
        )
        values["hunyuan_secret_id"] = get_from_dict_or_env(
            values,
            "hunyuan_secret_id",
            "HUNYUAN_SECRET_ID",
        )
        values["hunyuan_secret_key"] = _to_secret(
            get_from_dict_or_env(
                values,
                "hunyuan_secret_key",
                "HUNYUAN_SECRET_KEY",
            )
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Hunyuan API."""
        normal_params = {
            "app_id": self.hunyuan_app_id,
            "secret_id": self.hunyuan_secret_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if self.query_id is not None:
            normal_params["query_id"] = self.query_id

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
            return _generate_from_stream(stream_iter)

        res = self._chat(messages, **kwargs)

        response = res.json()

        if "error" in response:
            raise ValueError(f"Error from Hunyuan api response: {response}")

        return _create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        res = self._chat(messages, **kwargs)

        default_chunk_class = AIMessageChunk
        for chunk in res.iter_lines():
            response = json.loads(chunk)
            if "error" in response:
                raise ValueError(f"Error from Hunyuan api response: {response}")

            for choice in response["choices"]:
                chunk = _convert_delta_to_message_chunk(
                    choice["delta"], default_chunk_class
                )
                default_chunk_class = chunk.__class__
                yield ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content)

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        if self.hunyuan_secret_key is None:
            raise ValueError("Hunyuan secret key is not set.")

        parameters = {**self._default_params, **kwargs}

        headers = parameters.pop("headers", {})
        timestamp = parameters.pop("timestamp", int(time.time()))
        expired = parameters.pop("expired", timestamp + 24 * 60 * 60)

        payload = {
            "timestamp": timestamp,
            "expired": expired,
            "messages": [_convert_message_to_dict(m) for m in messages],
            **parameters,
        }

        if self.streaming:
            payload["stream"] = 1

        url = self.hunyuan_api_base + DEFAULT_PATH

        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": _signature(
                    secret_key=self.hunyuan_secret_key, url=url, payload=payload
                ),
                **headers,
            },
            json=payload,
            stream=self.streaming,
        )
        return res

    @property
    def _llm_type(self) -> str:
        return "hunyuan-chat"
