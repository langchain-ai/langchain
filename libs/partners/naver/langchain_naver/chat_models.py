"""Naver chat models."""
import logging
import os
from typing import Any, AsyncIterator, Iterator, List, Optional, Dict, cast, Tuple

import httpx
from httpx_sse import connect_sse

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import LangSmithParams, BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com/testapp"

logger = logging.getLogger(__name__)


def _convert_message_to_naver_chat_message(
    message: BaseMessage,
) -> Dict:
    if isinstance(message, ChatMessage):
        return dict(role=message.role, content=message.content)
    elif isinstance(message, HumanMessage):
        return dict(role="user", content=message.content)
    elif isinstance(message, SystemMessage):
        return dict(role="system", content=message.content)
    elif isinstance(message, AIMessage):
        return dict(role="assistant", content=message.content)
    else:
        logger.warning(
            "FunctionMessage, ToolMessage not yet supported "
            "(https://api.ncloud-docs.com/docs/clovastudio-chatcompletions)"
        )
        raise ValueError(f"Got unknown type {message}")


def _convert_naver_chat_message_to_message(
    _message: Dict,
) -> BaseMessage:
    role = _message["role"]
    assert role in ("assistant", "system", "user"), f"Expected role to be 'assistant', 'system', 'user', got {role}"
    content = cast(str, _message["content"])
    additional_kwargs: Dict = {}

    if role == "user":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs,)
    elif role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs,)
    elif role == "assistant":
        return AIMessage(content=content, additional_kwargs=additional_kwargs,)
    else:
        assert True, f"Expected role to be 'assistant', 'system', 'user', got {role}"


def _raise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = response.read().decode("utf-8")
        raise httpx.HTTPStatusError(
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}",
            request=response.request,
            response=response,
        )


class ChatNaver(BaseChatModel):
    """`NCP ClovaStudio` Chat Completion API.

    following environment variables set or passed in constructor in lower case:
    - ``NCP_CLOVASTUDIO_API_KEY``
    - ``NCP_APIGW_API_KEY``
    
    Example:
        .. code-block:: python

            from langchain_core.messages import HumanMessage

            from langchain_naver import ChatNaver

            model = ChatNaver()
            model.invoke([HumanMessage(content="Come up with 10 names for a song about parrots.")])
    """  # noqa: E501

    client: httpx.Client = Field(default=None)  #: :meta private:
    async_client: httpx.AsyncClient = Field(default=None)  #: :meta private:

    model_name: str = Field(default="HCX-003", alias="model")

    ncp_clovastudio_api_key: Optional[SecretStr] = Field(default=None, alias="clovastudio_api_key")
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_KEY` if not provided."""

    ncp_apigw_api_key: Optional[SecretStr] = Field(default=None, alias="apigw_api_key")
    """Automatically inferred from env are `NCP_APIGW_API_KEY` if not provided."""

    base_url: Optional[str] = Field(default=DEFAULT_BASE_URL, alias="ncp_clovastudio_api_base_url")
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_BASE_URL` if not provided."""

    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_before: Optional[str] = None
    include_ai_filters: Optional[bool] = None
    seed: Optional[int] = None
    timeout: int = 60

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the API."""
        defaults = {
            "temperature": self.temperature,
            "topK": self.top_k,
            "topP": self.top_p,
            "repeatPenalty": self.repeat_penalty,
            "maxTokens": self.max_tokens,
            "stopBefore": self.stop_before,
            "includeAiFilters": self.include_ai_filters,
            "seed": self.seed,
        }
        filtered = {k: v for k, v in defaults.items() if v is not None}
        return filtered

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "ncp_clovastudio_api_key": "NCP_CLOVASTUDIO_API_KEY",
            "ncp_apigw_api_key": "NCP_APIGW_API_KEY",
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-naver"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "naver"
        return params

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the client."""
        return self._default_params

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if values["temperature"] is not None and not 0 < values["temperature"] <= 1:
            raise ValueError("temperature must be in the range (0.0, 1.0]")

        if values["top_k"] is not None and not 0 <= values["top_k"] <= 128:
            raise ValueError("top_k must be in the range [0, 128]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["repeat_penalty"] is not None and not 0 < values["repeat_penalty"] <= 10:
            raise ValueError("repeat_penalty must be in the range (0.0, 10]")

        if values["max_tokens"] is not None and not 0 <= values["max_tokens"] <= 4096:
            raise ValueError("max_tokens must be in the range [0, 4096]")

        if values["seed"] is not None and not 0 <= values["temperature"] <= 4294967295:
            raise ValueError("temperature must be in the range [0, 4294967295]")

        """Validate that api key and python package exists in environment."""
        values["ncp_clovastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "ncp_clovastudio_api_key", "NCP_CLOVASTUDIO_API_KEY")
        )
        values["ncp_apigw_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "ncp_apigw_api_key", "NCP_APIGW_API_KEY")
        )
        values["base_url"] = get_from_dict_or_env(values, "base_url", "NCP_CLOVASTUDIO_API_BASE_URL")

        if not values.get("client"):
            values["client"] = httpx.Client(
                base_url=values["base_url"],
                headers=cls.default_headers(values),
                timeout=values["timeout"],
            )
        if not values.get("async_client"):
            values["async_client"] = httpx.AsyncClient(
                base_url=values["base_url"],
                headers=cls.default_headers(values),
                timeout=values["timeout"],
            )
        return values

    @staticmethod
    def default_headers(values):
        clovastudio_api_key = values["ncp_clovastudio_api_key"].get_secret_value() \
            if values["ncp_clovastudio_api_key"] \
            else None
        apigw_api_key = values["ncp_apigw_api_key"].get_secret_value() \
            if values["ncp_apigw_api_key"] \
            else None
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-NCP-CLOVASTUDIO-API-KEY": clovastudio_api_key,
            "X-NCP-APIGW-API-KEY": apigw_api_key,
        }

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        params = self._client_params
        if stop is not None and "stopBefore" in params:
            params["stopBefore"] = stop

        message_dicts = [_convert_message_to_naver_chat_message(m) for m in messages]
        return message_dicts, params

    def _completion_with_retry(self, **kwargs: Any) -> Any:
        if "stream" not in kwargs:
            kwargs["stream"] = False

        stream = kwargs["stream"]
        if stream:

            def iter_sse() -> Iterator[Dict]:
                with connect_sse(
                    self.client, "POST", f"/v1/chat-completions/{self.model_name}", json=kwargs
                ) as event_source:
                    _raise_on_error(event_source.response)
                    for event in event_source.iter_sse():
                        if event.data == "[DONE]":
                            return
                        yield event.json()

            return iter_sse()
        else:
            response = self.client.post(url=f"/v1/chat-completions/{self.model_name}", json=kwargs)
            _raise_on_error(response)
            return response.json()

    def _create_chat_result(self, response: Dict) -> ChatResult:
        generations = []
        result = response.get("result", {})
        msg = result.get("message", {})
        message = _convert_naver_chat_message_to_message(msg)
        gen = ChatGeneration(
            message=message,
        )
        generations.append(gen)

        llm_output = {
            "stop_reason": result.get("stopReason"),
            "input_length": result.get("inputLength"),
            "output_length": result.get("outputLength"),
            "seed": result.get("seed"),
            "ai_filter": result.get("aiFilter"),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self._completion_with_retry(
            messages=message_dicts, **params
        )
        return self._create_chat_result(response)

    # TODO: Implement if ChatNaver supports streaming. Otherwise delete method.
    # def _stream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[ChatGenerationChunk]:

    # TODO: Implement if ChatNaver supports async streaming. Otherwise delete.
    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:

    # TODO: Implement if ChatNaver supports async generation. Otherwise delete.
    # async def _agenerate(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> ChatResult:
