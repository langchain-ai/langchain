from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
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
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_community.llms.gigachat import _BaseGigaChat

if TYPE_CHECKING:
    import gigachat.models as gm

logger = logging.getLogger(__name__)


def _convert_dict_to_message(message: gm.Messages) -> BaseMessage:
    from gigachat.models import FunctionCall, MessagesRole

    additional_kwargs: Dict = {}
    if function_call := message.function_call:
        if isinstance(function_call, FunctionCall):
            additional_kwargs["function_call"] = dict(function_call)
        elif isinstance(function_call, dict):
            additional_kwargs["function_call"] = function_call

    if message.role == MessagesRole.SYSTEM:
        return SystemMessage(content=message.content)
    elif message.role == MessagesRole.USER:
        return HumanMessage(content=message.content)
    elif message.role == MessagesRole.ASSISTANT:
        return AIMessage(content=message.content, additional_kwargs=additional_kwargs)
    else:
        raise TypeError(f"Got unknown role {message.role} {message}")


def _convert_message_to_dict(message: gm.BaseMessage) -> gm.Messages:
    from gigachat.models import Messages, MessagesRole

    if isinstance(message, SystemMessage):
        return Messages(role=MessagesRole.SYSTEM, content=message.content)
    elif isinstance(message, HumanMessage):
        return Messages(role=MessagesRole.USER, content=message.content)
    elif isinstance(message, AIMessage):
        return Messages(
            role=MessagesRole.ASSISTANT,
            content=message.content,
            function_call=message.additional_kwargs.get("function_call", None),
        )
    elif isinstance(message, ChatMessage):
        return Messages(role=MessagesRole(message.role), content=message.content)
    elif isinstance(message, FunctionMessage):
        return Messages(role=MessagesRole.FUNCTION, content=message.content)
    else:
        raise TypeError(f"Got unknown type {message}")


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


class GigaChat(_BaseGigaChat, BaseChatModel):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import GigaChat
            giga = GigaChat(credentials=..., scope=..., verify_ssl_certs=False)
    """

    def _build_payload(self, messages: List[BaseMessage], **kwargs: Any) -> gm.Chat:
        from gigachat.models import Chat

        payload = Chat(
            messages=[_convert_message_to_dict(m) for m in messages],
        )

        payload.functions = kwargs.get("functions", None)
        payload.model = self.model

        if self.profanity_check is not None:
            payload.profanity_check = self.profanity_check
        if self.temperature is not None:
            payload.temperature = self.temperature
        if self.top_p is not None:
            payload.top_p = self.top_p
        if self.max_tokens is not None:
            payload.max_tokens = self.max_tokens
        if self.repetition_penalty is not None:
            payload.repetition_penalty = self.repetition_penalty
        if self.update_interval is not None:
            payload.update_interval = self.update_interval

        if self.verbose:
            logger.warning("Giga request: %s", payload.dict())

        return payload

    def _create_chat_result(self, response: Any) -> ChatResult:
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
            return generate_from_stream(stream_iter)

        payload = self._build_payload(messages, **kwargs)
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
            return await agenerate_from_stream(stream_iter)

        payload = self._build_payload(messages, **kwargs)
        response = await self._client.achat(payload)

        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, **kwargs)

        for chunk in self._client.stream(payload):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue

            choice = chunk["choices"][0]
            content = choice.get("delta", {}).get("content", {})
            chunk = _convert_delta_to_message_chunk(choice["delta"], AIMessageChunk)

            finish_reason = choice.get("finish_reason")

            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )

            if run_manager:
                run_manager.on_llm_new_token(content)

            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, **kwargs)

        async for chunk in self._client.astream(payload):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue

            choice = chunk["choices"][0]
            content = choice.get("delta", {}).get("content", {})
            chunk = _convert_delta_to_message_chunk(choice["delta"], AIMessageChunk)

            finish_reason = choice.get("finish_reason")

            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )

            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                await run_manager.on_llm_new_token(content)
