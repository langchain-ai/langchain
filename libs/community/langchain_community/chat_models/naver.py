import logging
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import httpx
from httpx_sse import SSEError
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.language_models.llms import create_base_retry_decorator
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
from langchain_core.utils import convert_to_secret_str, get_from_env
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

_DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com"

logger = logging.getLogger(__name__)


def _convert_chunk_to_message_chunk(
    sse: Any, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    sse_data = sse.json()
    if sse.event == "result":
        response_metadata = _sse_data_to_response_metadata(sse_data)
        return AIMessageChunk(content="", response_metadata=response_metadata)

    message = sse_data.get("message")
    role = message.get("role")
    content = message.get("content") or ""
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _sse_data_to_response_metadata(sse_data: Dict) -> Dict[str, Any]:
    response_metadata = {}
    if "stopReason" in sse_data:
        response_metadata["stop_reason"] = sse_data["stopReason"]
    if "inputLength" in sse_data:
        response_metadata["input_length"] = sse_data["inputLength"]
    if "outputLength" in sse_data:
        response_metadata["output_length"] = sse_data["outputLength"]
    if "seed" in sse_data:
        response_metadata["seed"] = sse_data["seed"]
    if "aiFilter" in sse_data:
        response_metadata["ai_filter"] = sse_data["aiFilter"]
    return response_metadata


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
    assert role in (
        "assistant",
        "system",
        "user",
    ), f"Expected role to be 'assistant', 'system', 'user', got {role}"
    content = cast(str, _message["content"])
    additional_kwargs: Dict = {}

    if role == "user":
        return HumanMessage(
            content=content,
            additional_kwargs=additional_kwargs,
        )
    elif role == "system":
        return SystemMessage(
            content=content,
            additional_kwargs=additional_kwargs,
        )
    elif role == "assistant":
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
        )
    else:
        logger.warning("Got unknown role %s", role)
        raise ValueError(f"Got unknown role {role}")


async def _aiter_sse(
    event_source_mgr: AsyncContextManager[Any],
) -> AsyncIterator[Dict]:
    """Iterate over the server-sent events."""
    async with event_source_mgr as event_source:
        await _araise_on_error(event_source.response)
        async for sse in event_source.aiter_sse():
            event_data = sse.json()
            if sse.event == "signal" and event_data.get("data", {}) == "[DONE]":
                return
            if sse.event == "error":
                raise SSEError(message=sse.data)
            yield sse


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


async def _araise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = (await response.aread()).decode("utf-8")
        raise httpx.HTTPStatusError(
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}",
            request=response.request,
            response=response,
        )


class ChatClovaX(BaseChatModel):
    """`NCP ClovaStudio` Chat Completion API.

    following environment variables set or passed in constructor in lower case:
    - ``NCP_CLOVASTUDIO_API_KEY``
    - ``NCP_APIGW_API_KEY``

    Example:
        .. code-block:: python

            from langchain_core.messages import HumanMessage

            from langchain_community import ChatClovaX

            model = ChatClovaX()
            model.invoke([HumanMessage(content="Come up with 10 names for a song about parrots.")])
    """  # noqa: E501

    client: Optional[httpx.Client] = Field(default=None)  #: :meta private:
    async_client: Optional[httpx.AsyncClient] = Field(default=None)  #: :meta private:

    model_name: str = Field(
        default="HCX-003",
        validation_alias=AliasChoices("model_name", "model"),
        description="NCP ClovaStudio chat model name",
    )
    task_id: Optional[str] = Field(
        default=None, description="NCP Clova Studio chat model tuning task ID"
    )
    service_app: bool = Field(
        default=False,
        description="false: use testapp, true: use service app on NCP Clova Studio",
    )

    ncp_clovastudio_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_KEY` if not provided."""

    ncp_apigw_api_key: Optional[SecretStr] = Field(default=None, alias="apigw_api_key")
    """Automatically inferred from env are `NCP_APIGW_API_KEY` if not provided."""

    base_url: str = Field(default="", alias="base_url")
    """
    Automatically inferred from env are `NCP_CLOVASTUDIO_API_BASE_URL` if not provided.
    """

    temperature: Optional[float] = Field(gt=0.0, le=1.0, default=0.5)
    top_k: Optional[int] = Field(ge=0, le=128, default=0)
    top_p: Optional[float] = Field(ge=0, le=1.0, default=0.8)
    repeat_penalty: Optional[float] = Field(gt=0.0, le=10, default=5.0)
    max_tokens: Optional[int] = Field(ge=0, le=4096, default=100)
    stop_before: Optional[list[str]] = Field(default=None, alias="stop")
    include_ai_filters: Optional[bool] = Field(default=False)
    seed: Optional[int] = Field(ge=0, le=4294967295, default=0)

    timeout: int = Field(gt=0, default=90)
    max_retries: int = Field(ge=1, default=2)

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

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
        self._default_params["model_name"] = self.model_name
        return self._default_params

    @property
    def lc_secrets(self) -> Dict[str, str]:
        if not self._is_new_api_key():
            return {
                "ncp_clovastudio_api_key": "NCP_CLOVASTUDIO_API_KEY",
            }
        else:
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

    @property
    def _api_url(self) -> str:
        """GET chat completion api url"""
        app_type = "serviceapp" if self.service_app else "testapp"

        if self.task_id:
            return (
                f"{self.base_url}/{app_type}/v1/tasks/{self.task_id}/chat-completions"
            )
        else:
            return f"{self.base_url}/{app_type}/v1/chat-completions/{self.model_name}"

    @model_validator(mode="after")
    def validate_model_after(self) -> Self:
        if not (self.model_name or self.task_id):
            raise ValueError("either model_name or task_id must be assigned a value.")

        if not self.ncp_clovastudio_api_key:
            self.ncp_clovastudio_api_key = convert_to_secret_str(
                get_from_env("ncp_clovastudio_api_key", "NCP_CLOVASTUDIO_API_KEY")
            )

        if not self._is_new_api_key():
            self._init_fields_on_old_api_key()

        if not self.base_url:
            self.base_url = get_from_env(
                "base_url", "NCP_CLOVASTUDIO_API_BASE_URL", _DEFAULT_BASE_URL
            )

        if not self.client:
            self.client = httpx.Client(
                base_url=self.base_url,
                headers=self.default_headers(),
                timeout=self.timeout,
            )

        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.default_headers(),
                timeout=self.timeout,
            )

        return self

    def _is_new_api_key(self) -> bool:
        if self.ncp_clovastudio_api_key:
            return self.ncp_clovastudio_api_key.get_secret_value().startswith("nv-")
        else:
            return False

    def _init_fields_on_old_api_key(self) -> None:
        if not self.ncp_apigw_api_key:
            self.ncp_apigw_api_key = convert_to_secret_str(
                get_from_env("ncp_apigw_api_key", "NCP_APIGW_API_KEY", "")
            )

    def default_headers(self) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        clovastudio_api_key = (
            self.ncp_clovastudio_api_key.get_secret_value()
            if self.ncp_clovastudio_api_key
            else None
        )

        if self._is_new_api_key():
            ### headers on new api key
            headers["Authorization"] = f"Bearer {clovastudio_api_key}"
        else:
            ### headers on old api key
            if clovastudio_api_key:
                headers["X-NCP-CLOVASTUDIO-API-KEY"] = clovastudio_api_key

            apigw_api_key = (
                self.ncp_apigw_api_key.get_secret_value()
                if self.ncp_apigw_api_key
                else None
            )
            if apigw_api_key:
                headers["X-NCP-APIGW-API-KEY"] = apigw_api_key

        return headers

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        params = self._client_params
        if stop is not None and "stopBefore" in params:
            params["stopBefore"] = stop

        message_dicts = [_convert_message_to_naver_chat_message(m) for m in messages]
        return message_dicts, params

    def _completion_with_retry(self, **kwargs: Any) -> Any:
        from httpx_sse import (
            ServerSentEvent,
            connect_sse,
        )

        if "stream" not in kwargs:
            kwargs["stream"] = False

        stream = kwargs["stream"]
        client = cast(httpx.Client, self.client)
        if stream:

            def iter_sse() -> Iterator[ServerSentEvent]:
                with connect_sse(
                    client, "POST", self._api_url, json=kwargs
                ) as event_source:
                    _raise_on_error(event_source.response)
                    for sse in event_source.iter_sse():
                        event_data = sse.json()
                        if (
                            sse.event == "signal"
                            and event_data.get("data", {}) == "[DONE]"
                        ):
                            return
                        if sse.event == "error":
                            raise SSEError(message=sse.data)
                        yield sse

            return iter_sse()
        else:
            response = client.post(url=self._api_url, json=kwargs)
            _raise_on_error(response)
            return response.json()

    async def _acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        from httpx_sse import aconnect_sse

        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            if "stream" not in kwargs:
                kwargs["stream"] = False
            stream = kwargs["stream"]
            async_client = cast(httpx.AsyncClient, self.async_client)
            if stream:
                event_source = aconnect_sse(
                    async_client, "POST", self._api_url, json=kwargs
                )
                return _aiter_sse(event_source)
            else:
                response = await async_client.post(url=self._api_url, json=kwargs)
                await _araise_on_error(response)
                return response.json()

        return await _completion_with_retry(**kwargs)

    def _create_chat_result(self, response: Dict) -> ChatResult:
        generations = []
        result = response.get("result", {})
        msg = result.get("message", {})
        message = _convert_naver_chat_message_to_message(msg)

        if isinstance(message, AIMessage):
            message.usage_metadata = {
                "input_tokens": result.get("inputLength"),
                "output_tokens": result.get("outputLength"),
                "total_tokens": result.get("inputLength") + result.get("outputLength"),
            }

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

        response = self._completion_with_retry(messages=message_dicts, **params)

        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        for sse in self._completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            new_chunk = _convert_chunk_to_message_chunk(sse, default_chunk_class)
            default_chunk_class = new_chunk.__class__
            gen_chunk = ChatGenerationChunk(message=new_chunk)

            if run_manager:
                run_manager.on_llm_new_token(
                    token=cast(str, new_chunk.content), chunk=gen_chunk
                )

            yield gen_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        response = await self._acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )

        return self._create_chat_result(response)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        async for chunk in await self._acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            new_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            default_chunk_class = new_chunk.__class__
            gen_chunk = ChatGenerationChunk(message=new_chunk)

            if run_manager:
                await run_manager.on_llm_new_token(
                    token=cast(str, new_chunk.content), chunk=gen_chunk
                )

            yield gen_chunk


def _create_retry_decorator(
    llm: ChatClovaX,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle exceptions"""

    errors = [httpx.RequestError, httpx.StreamError]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
