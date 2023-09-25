"""
GigaChatModel for GigaChat.
"""
import logging
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
)

import gigachat
from gigachat.models import (
    ChatCompletion,
    MessagesRes,
    MessagesRole,
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import (
    BaseChatModel,
    _agenerate_from_stream,
    _generate_from_stream,
)
from langchain.schema import ChatResult
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import (
    ChatGeneration,
    ChatGenerationChunk,
)

logger = logging.getLogger(__name__)


def _convert_dict_to_message(message: MessagesRes) -> BaseMessage:
    if message.role == MessagesRole.SYSTEM:
        return SystemMessage(content=message.content)
    elif message.role == MessagesRole.USER:
        return HumanMessage(content=message.content)
    elif message.role == MessagesRole.ASSISTANT:
        return AIMessage(content=message.content)
    else:
        raise TypeError(f"Got unknown role {message.role} {message}")


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, str]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    elif isinstance(message, ChatMessage):
        if message.role not in [role for role in MessagesRole]:
            raise TypeError(f"Got unknown role {message.role} {message}")
        return {"role": message.role, "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")


class GigaChat(BaseChatModel):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain.chat_models import GigaChat
            giga = GigaChat(user="username", password="password")
    """

    use_auth: Optional[bool] = None
    api_base_url: Optional[str] = None
    token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[float] = None
    verify_ssl: Optional[bool] = None
    """ Check certificates for all requests """
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    oauth_base_url: Optional[str] = None
    oauth_token: Optional[str] = None
    oauth_scope: Optional[str] = None
    oauth_timeout: Optional[float] = None
    oauth_verify_ssl: Optional[bool] = None

    profanity: bool = True
    streaming: bool = False
    """ Whether to stream the results or not. """
    temperature: float = 0
    max_tokens: int = 0
    """ Maximum number of tokens to generate """

    @property
    def _llm_type(self) -> str:
        return "giga-chat-model"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "token": "GIGA_TOKEN",
            "password": "GIGA_PASSWORD",
            "client_secret": "GIGA_CLIENT_SECRET",
            "oauth_token": "GIGA_OAUTH_TOKEN",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    @cached_property
    def _client(self) -> gigachat.GigaChat:
        return gigachat.GigaChat(**self.__dict__)

    def _build_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [_convert_message_to_dict(m) for m in messages],
            "profanity_check": self.profanity,
        }
        if self.temperature > 0:
            payload["temperature"] = self.temperature
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        if self.verbose:
            logger.warning("Giga request: %s", payload)

        return payload

    def _create_chat_result(self, response: ChatCompletion) -> ChatResult:
        generations = []
        for res in response.choices:
            message = _convert_dict_to_message(res.message)
            finish_reason = res.finish_reason
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)
            if finish_reason != "stop":
                logger.warning(
                    "Giga generation stopped with reason: %s",
                    finish_reason,
                )
            if self.verbose:
                logger.warning("Giga response: %s", message.content)
        token_usage = response.usage
        llm_output = {"token_usage": token_usage, "model_name": response.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return _generate_from_stream(stream_iter)

        payload = self._build_payload(messages)
        response = self._client.chat(payload)

        return self._create_chat_result(response)

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
            return await _agenerate_from_stream(stream_iter)

        payload = self._build_payload(messages)
        response = await self._client.achat(payload)

        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages)

        for chunk in self._client.stream(payload):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager:
                    run_manager.on_llm_new_token(content)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages)

        async for chunk in self._client.astream(payload):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager:
                    await run_manager.on_llm_new_token(content)

    def get_num_tokens(self, text: str) -> int:
        return round(len(text) / 4.6)
