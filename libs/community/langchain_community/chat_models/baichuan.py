import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
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
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.baichuan-ai.com/v1/chat/completions"


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
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


class ChatBaichuan(BaseChatModel):
    """Baichuan chat models API by Baichuan Intelligent Technology.

    For more information, see https://platform.baichuan-ai.com/docs/api
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "baichuan_api_key": "BAICHUAN_API_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    baichuan_api_base: str = Field(default=DEFAULT_API_BASE)
    """Baichuan custom endpoints"""
    baichuan_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Baichuan API Key"""
    baichuan_secret_key: Optional[SecretStr] = None
    """[DEPRECATED, keeping it for for backward compatibility] Baichuan Secret Key"""
    streaming: bool = False
    """Whether to stream the results or not."""
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
    """Whether to use search enhance, default is False."""
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
        values["baichuan_api_base"] = get_from_dict_or_env(
            values,
            "baichuan_api_base",
            "BAICHUAN_API_BASE",
            DEFAULT_API_BASE,
        )
        values["baichuan_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "baichuan_api_key",
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
            "with_search_enhance": self.with_search_enhance,
            "stream": self.streaming,
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
        res = self._chat(messages, **kwargs)
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

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        parameters = {**self._default_params, **kwargs}

        model = parameters.pop("model")
        headers = parameters.pop("headers", {})
        temperature = parameters.pop("temperature", 0.3)
        top_k = parameters.pop("top_k", 5)
        top_p = parameters.pop("top_p", 0.85)
        with_search_enhance = parameters.pop("with_search_enhance", False)
        stream = parameters.pop("stream", False)

        payload = {
            "model": model,
            "messages": [_convert_message_to_dict(m) for m in messages],
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "with_search_enhance": with_search_enhance,
            "stream": stream,
        }

        url = self.baichuan_api_base
        api_key = ""
        if self.baichuan_api_key:
            api_key = self.baichuan_api_key.get_secret_value()

        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                **headers,
            },
            json=payload,
            stream=self.streaming,
        )
        return res

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
