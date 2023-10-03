# flake8: noqa: I001
"""
GigaChatModel for GigaChat.
"""
import logging
from functools import cached_property
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import gigachat
from gigachat.models import Chat, ChatCompletion, Messages, MessagesRole
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
from langchain.schema.output import ChatGeneration, ChatGenerationChunk

logger = logging.getLogger(__name__)


def _convert_dict_to_message(message: Messages) -> BaseMessage:
    if message.role == MessagesRole.SYSTEM:
        return SystemMessage(content=message.content)
    elif message.role == MessagesRole.USER:
        return HumanMessage(content=message.content)
    elif message.role == MessagesRole.ASSISTANT:
        return AIMessage(content=message.content)
    else:
        raise TypeError(f"Got unknown role {message.role} {message}")


def _convert_message_to_dict(message: BaseMessage) -> Messages:
    if isinstance(message, SystemMessage):
        return Messages(role=MessagesRole.SYSTEM, content=message.content)
    elif isinstance(message, HumanMessage):
        return Messages(role=MessagesRole.USER, content=message.content)
    elif isinstance(message, AIMessage):
        return Messages(role=MessagesRole.ASSISTANT, content=message.content)
    elif isinstance(message, ChatMessage):
        return Messages(role=MessagesRole(message.role), content=message.content)
    else:
        raise TypeError(f"Got unknown type {message}")


class GigaChat(BaseChatModel):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain.chat_models import GigaChat
            giga = GigaChat(credentials=..., verify_ssl_certs=False)
    """

    base_url: Optional[str] = None
    """ Адрес относительно которого выполняются запросы """
    auth_url: Optional[str] = None
    credentials: Optional[str] = None
    """ Авторизационные данные """
    scope: Optional[str] = None

    access_token: Optional[str] = None
    model: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    timeout: Optional[float] = None
    verify_ssl_certs: Optional[bool] = None
    """ Check certificates for all requests """

    use_auth: Optional[bool] = None

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
            "credentials": "GIGACHAT_CREDENTIALS",
            "access_token": "GIGACHAT_ACCESS_TOKEN",
            "password": "GIGACHAT_PASSWORD",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    @cached_property
    def _client(self) -> gigachat.GigaChat:
        return gigachat.GigaChat(
            base_url=self.base_url,
            auth_url=self.auth_url,
            credentials=self.credentials,
            scope=self.scope,
            access_token=self.access_token,
            model=self.model,
            user=self.user,
            password=self.password,
            timeout=self.timeout,
            verify_ssl_certs=self.verify_ssl_certs,
            use_auth=self.use_auth,
        )

    def _build_payload(self, messages: List[BaseMessage]) -> Chat:
        payload = Chat(
            messages=[_convert_message_to_dict(m) for m in messages],
            profanity_check=self.profanity,
        )
        if self.temperature > 0:
            payload.temperature = self.temperature
        if self.max_tokens > 0:
            payload.max_tokens = self.max_tokens

        if self.verbose:
            logger.warning("Giga request: %s", payload.dict())

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
        llm_output = {"token_usage": response.usage, "model_name": response.model}
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
