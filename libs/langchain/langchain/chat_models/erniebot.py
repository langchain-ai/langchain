from __future__ import annotations

import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import create_base_retry_decorator
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

_MessageDict = Dict[str, Any]


class ErnieBotChat(BaseChatModel):
    """ERNIE Bot Chat large language models API.

    To use, you should have the ``erniebot`` python package installed, and the
    environment variable ``EB_ACCESS_TOKEN`` set with your AI Studio access
    token.

    Example:
        .. code-block:: python
            from langchain.chat_models import ErnieBotChat
            erniebot_chat = ErnieBotChat(model="ernie-bot")
    """

    client: Any = None
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    aistudio_access_token: Optional[str] = None
    """AI Studio access token."""
    streaming: Optional[bool] = False
    """Whether to stream the results or not."""
    model: str = "ernie-bot"
    """Model to use."""
    top_p: Optional[float] = 0.8
    """Parameter of nucleus sampling that affects the diversity of generated content."""
    temperature: Optional[float] = 0.95
    """Sampling temperature to use."""
    penalty_score: Optional[float] = 1
    """Penalty assigned to tokens that have been generated."""
    request_timeout: Optional[int] = 60
    """How many seconds to wait for the server to send data before giving up."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    ernie_client_id: Optional[str] = None
    ernie_client_secret: Optional[str] = None
    """For raising deprecation warnings."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ERNIE Bot API."""
        normal_params = {
            "model": self.model,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
            "request_timeout": self.request_timeout,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self._default_params

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        auth_cfg: Dict[str, Optional[str]] = {
            "api_type": "aistudio",
            "access_token": self.aistudio_access_token,
        }
        return {**{"_config_": auth_cfg}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "erniebot"

    @root_validator()
    def validate_enviroment(cls, values: Dict) -> Dict:
        try:
            aistudio_access_token = get_from_dict_or_env(
                values,
                "aistudio_access_token",
                "EB_ACCESS_TOKEN",
            )
        except ValueError as e:
            if (
                "ernie_client_id" in values
                and values["ernie_client_id"]
                or "ernie_client_secret" in values
                and values["ernie_client_secret"]
                or "ERNIE_CLIENT_ID" in os.environ
                or "ERNIE_CLIENT_SECRET" in os.environ
            ):
                raise RuntimeError(
                    "The authentication parameters "
                    "`ernie_client_id` and `ernie_client_secret` are deprecated. "
                    "For AI Studio users, please set "
                    "`aistudio_access_token` to your AI Studio access token. "
                    "For Qianfan users, please use "
                    "`langchain.chat_models.QianfanChatEndpoint` instead."
                ) from e
            else:
                raise
        else:
            values["aistudio_access_token"] = aistudio_access_token

        try:
            import erniebot

            values["client"] = erniebot.ChatCompletion
        except ImportError:
            raise ImportError(
                "Could not import erniebot python package. "
                "Please install it with `pip install erniebot`."
            )
        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            chunks = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            generation: Optional[ChatGenerationChunk] = None
            for chunk in chunks:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        else:
            params = self._invocation_params
            params.update(kwargs)
            params["messages"] = self._convert_messages_to_dicts(messages)
            params["stream"] = False
            response = _create_completion_with_retry(
                self, run_manager=run_manager, **params
            )
            return self._build_chat_result_from_response(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            chunks = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            generation: Optional[ChatGenerationChunk] = None
            async for chunk in chunks:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        else:
            params = self._invocation_params
            params.update(kwargs)
            params["messages"] = self._convert_messages_to_dicts(messages)
            params["stream"] = False
            response = await _acreate_completion_with_retry(
                self, run_manager=run_manager, **params
            )
            return self._build_chat_result_from_response(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stop is not None:
            raise TypeError(
                "Currently, `stop` is not supported when streaming is enabled."
            )
        params = self._invocation_params
        params.update(kwargs)
        params["messages"] = self._convert_messages_to_dicts(messages)
        params["stream"] = True
        for resp in _create_completion_with_retry(
            self, run_manager=run_manager, **params
        ):
            chunk = self._build_chunk_from_response(resp)
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
        if stop is not None:
            raise TypeError(
                "Currently, `stop` is not supported when streaming is enabled."
            )
        params = self._invocation_params
        params.update(kwargs)
        params["messages"] = self._convert_messages_to_dicts(messages)
        params["stream"] = True
        async for resp in await _acreate_completion_with_retry(
            self, run_manager=run_manager, **params
        ):
            chunk = self._build_chunk_from_response(resp)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _build_chat_result_from_response(
        self, response: Mapping[str, Any]
    ) -> ChatResult:
        message_dict = self._build_dict_from_response(response)
        generation = ChatGeneration(
            message=self._convert_dict_to_message(message_dict),
            generation_info=dict(finish_reason="stop"),
        )
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return ChatResult(generations=[generation], llm_output=llm_output)

    def _build_chunk_from_response(
        self, response: Mapping[str, Any]
    ) -> ChatGenerationChunk:
        message_dict = self._build_dict_from_response(response)
        message = self._convert_dict_to_message(message_dict)
        msg_chunk = AIMessageChunk(
            content=message.content,
            additional_kwargs=message.additional_kwargs,
        )
        return ChatGenerationChunk(message=msg_chunk)

    def _build_dict_from_response(self, response: Mapping[str, Any]) -> _MessageDict:
        message_dict: _MessageDict = {"role": "assistant"}
        if "function_call" in response:
            message_dict["content"] = None
            message_dict["function_call"] = response["function_call"]
        else:
            message_dict["content"] = response["result"]
        return message_dict

    def _convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[dict]:
        erniebot_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                logger.warning(
                    "Ignoring system messages "
                    "since they are currently not supported for ERNIE Bot."
                )
            eb_msg = self._convert_message_to_dict(msg)
            erniebot_messages.append(eb_msg)
        return erniebot_messages

    @staticmethod
    def _convert_dict_to_message(message_dict: _MessageDict) -> BaseMessage:
        role = message_dict["role"]
        if role == "user":
            return HumanMessage(content=message_dict["content"])
        elif role == "assistant":
            content = message_dict.get("content", "")
            if message_dict.get("function_call"):
                additional_kwargs = {
                    "function_call": dict(message_dict["function_call"])
                }
            else:
                additional_kwargs = {}
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif role == "function":
            return FunctionMessage(
                content=message_dict["content"], name=message_dict["name"]
            )
        else:
            return ChatMessage(content=message_dict["content"], role=role)

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> _MessageDict:
        message_dict: _MessageDict
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs[
                    "function_call"
                ]
                if message_dict["content"] == "":
                    message_dict["content"] = None
        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }
        else:
            raise TypeError(f"Got unknown type {message}")

        return message_dict


def _create_completion_with_retry(
    llm: ErnieBotChat,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _client_create(**kwargs: Any) -> Any:
        return llm.client.create(**kwargs)

    return _client_create(**kwargs)


async def _acreate_completion_with_retry(
    llm: ErnieBotChat,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _client_acreate(**kwargs: Any) -> Any:
        return await llm.client.acreate(**kwargs)

    return await _client_acreate(**kwargs)


def _create_retry_decorator(
    llm: ErnieBotChat,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import erniebot

    errors: List[Type[BaseException]] = [
        erniebot.errors.TimeoutError,
        erniebot.errors.RequestLimitError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
