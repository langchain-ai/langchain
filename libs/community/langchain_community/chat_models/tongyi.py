from __future__ import annotations

import asyncio
import functools
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
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
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain_community.llms.tongyi import check_response

logger = logging.getLogger(__name__)


def convert_dict_to_message(
    _dict: Mapping[str, Any], is_chunk: bool = False
) -> Union[BaseMessage, BaseMessageChunk]:
    role = _dict["role"]
    content = _dict["content"]
    if role == "user":
        return (
            HumanMessageChunk(content=content)
            if is_chunk
            else HumanMessage(content=content)
        )
    elif role == "assistant":
        return (
            AIMessageChunk(content=content) if is_chunk else AIMessage(content=content)
        )
    elif role == "system":
        return (
            SystemMessageChunk(content=content)
            if is_chunk
            else SystemMessage(content=content)
        )
    else:
        return (
            ChatMessageChunk(role=role, content=content)
            if is_chunk
            else ChatMessage(role=role, content=content)
        )


def convert_message_chunk_to_message(message_chunk: BaseMessageChunk) -> BaseMessage:
    if isinstance(message_chunk, HumanMessageChunk):
        return HumanMessage(content=message_chunk.content)
    elif isinstance(message_chunk, AIMessageChunk):
        return AIMessage(content=message_chunk.content)
    elif isinstance(message_chunk, SystemMessageChunk):
        return SystemMessage(content=message_chunk.content)
    elif isinstance(message_chunk, ChatMessageChunk):
        return ChatMessage(role=message_chunk.role, content=message_chunk.content)
    else:
        raise TypeError(f"Got unknown type {message_chunk}")


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a message to a dict."""

    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
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
    """Alibaba Tongyi Qwen chat models API.

    To use, you should have the ``dashscope`` python package installed,
    and set env ``DASHSCOPE_API_KEY`` with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import Tongyi
            Tongyi_chat = ChatTongyi()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    client: Any  #: :meta private:
    model_name: str = Field(default="qwen-turbo", alias="model")

    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[SecretStr] = None
    """Dashscope api key provide by Alibaba Cloud."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @root_validator()
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
            "api_key": self.dashscope_api_key.get_secret_value(),
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
            for resp in responses:
                yield check_response(resp)

        return _stream_completion_with_retry(**kwargs)

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
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(
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
                functools.partial(
                    self.completion_with_retry, **{"run_manager": run_manager, **params}
                ),
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
        for stream_resp in self.stream_completion_with_retry(**params):
            chunk = ChatGenerationChunk(
                **self._chat_generation_from_qwen_resp(stream_resp, is_chunk=True)
            )
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

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
        async for stream_resp in self.astream_completion_with_retry(**params):
            chunk = ChatGenerationChunk(
                **self._chat_generation_from_qwen_resp(stream_resp, is_chunk=True)
            )
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _invocation_params(
        self, messages: List[BaseMessage], stop: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        params = {**self._default_params, **kwargs}
        if stop is not None:
            params["stop"] = stop
        if params.get("stream"):
            params["incremental_output"] = True

        message_dicts = [convert_message_to_dict(m) for m in messages]

        # According to the docs, the last message should be a `user` message
        if message_dicts[-1]["role"] != "user":
            raise ValueError("Last message should be user message.")
        # And the `system` message should be the first message if present
        system_message_indices = [
            i for i, m in enumerate(message_dicts) if m["role"] == "system"
        ]
        if len(system_message_indices) != 1 or system_message_indices[0] != 0:
            raise ValueError("System message can only be the first message.")

        params["messages"] = message_dicts

        return params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        if llm_outputs[0] is None:
            return {}
        return llm_outputs[0]

    @staticmethod
    def _chat_generation_from_qwen_resp(
        resp: Any, is_chunk: bool = False
    ) -> Dict[str, Any]:
        choice = resp["output"]["choices"][0]
        message = convert_dict_to_message(choice["message"], is_chunk=is_chunk)
        return dict(
            message=message,
            generation_info=dict(
                finish_reason=choice["finish_reason"],
                request_id=resp["request_id"],
                token_usage=dict(resp["usage"]),
            ),
        )

    @staticmethod
    def _chunk_to_generation(chunk: ChatGenerationChunk) -> ChatGeneration:
        return ChatGeneration(
            message=convert_message_chunk_to_message(chunk.message),
            generation_info=chunk.generation_info,
        )
