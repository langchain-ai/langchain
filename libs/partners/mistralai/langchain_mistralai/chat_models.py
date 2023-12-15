from __future__ import annotations

import importlib.util
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
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
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env

# TODO: Remove 'type: ignore' once mistralai has stubs or py.typed marker.
from mistralai.async_client import MistralAsyncClient  # type: ignore[import]
from mistralai.client import MistralClient  # type: ignore[import]
from mistralai.constants import (  # type: ignore[import]
    ENDPOINT as DEFAULT_MISTRAL_ENDPOINT,
)
from mistralai.exceptions import (  # type: ignore[import]
    MistralAPIException,
    MistralConnectionException,
    MistralException,
)
from mistralai.models.chat_completion import (  # type: ignore[import]
    ChatCompletionResponse as MistralChatCompletionResponse,
)
from mistralai.models.chat_completion import (  # type: ignore[import]
    ChatMessage as MistralChatMessage,
)
from mistralai.models.chat_completion import (  # type: ignore[import]
    DeltaMessage as MistralDeltaMessage,
)

logger = logging.getLogger(__name__)


def _create_retry_decorator(
    llm: ChatMistralAI,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle exceptions"""

    errors = [
        MistralException,
        MistralAPIException,
        MistralConnectionException,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


def _convert_mistral_chat_message_to_message(
    _message: MistralChatMessage,
) -> BaseMessage:
    role = _message.role
    if role == "user":
        return HumanMessage(content=_message.content)
    elif role == "assistant":
        return AIMessage(content=_message.content)
    elif role == "system":
        return SystemMessage(content=_message.content)
    else:
        return ChatMessage(content=_message.content, role=role)


async def acompletion_with_retry(
    llm: ChatMistralAI,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        client = MistralAsyncClient(
            api_key=llm.mistral_api_key,
            endpoint=llm.endpoint,
            max_retries=llm.max_retries,
            timeout=llm.timeout,
            max_concurrent_requests=llm.max_concurrent_requests,
        )
        stream = kwargs.pop("stream", False)
        if stream:
            return client.chat_stream(**kwargs)
        else:
            return await client.chat(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_delta_to_message_chunk(
    _obj: MistralDeltaMessage, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = getattr(_obj, "role")
    content = getattr(_obj, "content", "")
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


def _convert_message_to_mistral_chat_message(
    message: BaseMessage,
) -> MistralChatMessage:
    if isinstance(message, ChatMessage):
        mistral_message = MistralChatMessage(role=message.role, content=message.content)
    elif isinstance(message, HumanMessage):
        mistral_message = MistralChatMessage(role="user", content=message.content)
    elif isinstance(message, AIMessage):
        mistral_message = MistralChatMessage(role="assistant", content=message.content)
    elif isinstance(message, SystemMessage):
        mistral_message = MistralChatMessage(role="system", content=message.content)
    else:
        raise ValueError(f"Got unknown type {message}")
    return mistral_message


class ChatMistralAI(BaseChatModel):
    """A chat model that uses the MistralAI API."""

    client: Any  #: :meta private:
    mistral_api_key: Optional[str] = None
    endpoint: str = DEFAULT_MISTRAL_ENDPOINT
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 64

    model: str = "mistral-small"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    random_seed: Optional[int] = None
    safe_mode: bool = False

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the API."""
        defaults = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "random_seed": self.random_seed,
            "safe_mode": self.safe_mode,
        }
        filtered = {k: v for k, v in defaults.items() if v is not None}
        return filtered

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the client."""
        return self._default_params

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            stream = kwargs.pop("stream", False)
            if stream:
                return self.client.chat_stream(**kwargs)
            else:
                return self.client.chat(**kwargs)

        return _completion_with_retry(**kwargs)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists, temperature, and top_p."""
        mistralai_spec = importlib.util.find_spec("mistralai")
        if mistralai_spec is None:
            raise MistralException(
                "Could not find mistralai python package. "
                "Please install it with `pip install mistralai`"
            )

        values["mistral_api_key"] = get_from_dict_or_env(
            values, "mistral_api_key", "MISTRAL_API_KEY", default=""
        )
        values["client"] = MistralClient(
            api_key=values["mistral_api_key"],
            endpoint=values["endpoint"],
            max_retries=values["max_retries"],
            timeout=values["timeout"],
        )

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _create_chat_result(
        self, response: MistralChatCompletionResponse
    ) -> ChatResult:
        generations = []
        for res in response.choices:
            finish_reason = getattr(res, "finish_reason")
            if finish_reason:
                finish_reason = finish_reason.value
            gen = ChatGeneration(
                message=_convert_mistral_chat_message_to_message(res.message),
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)
        token_usage = getattr(response, "usage")
        token_usage = vars(token_usage) if token_usage else {}
        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[MistralChatMessage], Dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_mistral_chat_message(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mistralai-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"mistral_api_key": "MISTRAL_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "mistralai"]
