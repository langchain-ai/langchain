from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Union, cast, overload

from ai21.models import ChatMessage as J2ChatMessage
from ai21.models import RoleType
from ai21.models.chat import ChatCompletionChunk, ChatMessage
from ai21.stream.stream import Stream as AI21Stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk

_ChatMessageTypes = Union[ChatMessage, J2ChatMessage]
_SYSTEM_ERR_MESSAGE = "System message must be at beginning of message list."
_ROLE_TYPE = Union[str, RoleType]


class ChatAdapter(ABC):
    """Common interface for the different Chat models available in AI21.

    It converts LangChain messages to AI21 messages.
    Calls the appropriate AI21 model API with the converted messages.
    """

    @abstractmethod
    def convert_messages(
        self,
        messages: List[BaseMessage],
    ) -> Dict[str, Any]:
        pass

    def _convert_message_to_ai21_message(
        self,
        message: BaseMessage,
    ) -> _ChatMessageTypes:
        content = cast(str, message.content)
        role = self._parse_role(message)

        return self._chat_message(role=role, content=content)

    def _parse_role(self, message: BaseMessage) -> _ROLE_TYPE:
        role = None

        if isinstance(message, HumanMessage):
            return RoleType.USER

        if isinstance(message, AIMessage):
            return RoleType.ASSISTANT

        if isinstance(self, J2ChatAdapter):
            if not role:
                raise ValueError(
                    f"Could not resolve role type from message {message}. "
                    f"Only support {HumanMessage.__name__} and {AIMessage.__name__}."
                )

        # if it gets here, we rely on the server to handle the role type
        return message.type

    @abstractmethod
    def _chat_message(
        self,
        role: _ROLE_TYPE,
        content: str,
    ) -> _ChatMessageTypes:
        pass

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[True],
        **params: Any,
    ) -> Iterator[ChatGenerationChunk]:
        pass

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[False],
        **params: Any,
    ) -> List[BaseMessage]:
        pass

    @abstractmethod
    def call(
        self,
        client: Any,
        stream: Literal[True] | Literal[False],
        **params: Any,
    ) -> List[BaseMessage] | Iterator[ChatGenerationChunk]:
        pass

    def _get_system_message_from_message(self, message: BaseMessage) -> str:
        if not isinstance(message.content, str):
            raise ValueError(
                f"System Message must be of type str. Got {type(message.content)}"
            )

        return message.content


class J2ChatAdapter(ChatAdapter):
    """Adapter for J2Chat models."""

    def convert_messages(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        system_message = ""
        converted_messages = []  # type: ignore

        for i, message in enumerate(messages):
            if message.type == "system":
                if i != 0:
                    raise ValueError(_SYSTEM_ERR_MESSAGE)
                else:
                    system_message = self._get_system_message_from_message(message)
            else:
                converted_message = self._convert_message_to_ai21_message(message)
                converted_messages.append(converted_message)

        return {"system": system_message, "messages": converted_messages}

    def _chat_message(
        self,
        role: _ROLE_TYPE,
        content: str,
    ) -> J2ChatMessage:
        return J2ChatMessage(role=RoleType(role), text=content)

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[True],
        **params: Any,
    ) -> Iterator[ChatGenerationChunk]:
        ...

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[False],
        **params: Any,
    ) -> List[BaseMessage]:
        ...

    def call(
        self,
        client: Any,
        stream: Literal[True] | Literal[False],
        **params: Any,
    ) -> List[BaseMessage] | Iterator[ChatGenerationChunk]:
        if stream:
            raise NotImplementedError("Streaming is not supported for Jurassic models.")

        response = client.chat.create(**params)

        return [AIMessage(output.text) for output in response.outputs]


class JambaChatCompletionsAdapter(ChatAdapter):
    """Adapter for Jamba Chat Completions."""

    def convert_messages(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        return {
            "messages": [
                self._convert_message_to_ai21_message(message) for message in messages
            ],
        }

    def _chat_message(
        self,
        role: _ROLE_TYPE,
        content: str,
    ) -> ChatMessage:
        return ChatMessage(
            role=role.value if isinstance(role, RoleType) else role,
            content=content,
        )

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[True],
        **params: Any,
    ) -> Iterator[ChatGenerationChunk]:
        ...

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[False],
        **params: Any,
    ) -> List[BaseMessage]:
        ...

    def call(
        self,
        client: Any,
        stream: Literal[True] | Literal[False],
        **params: Any,
    ) -> List[BaseMessage] | Iterator[ChatGenerationChunk]:
        response = client.chat.completions.create(stream=stream, **params)

        if stream:
            return self._stream_response(response)

        return [AIMessage(choice.message.content) for choice in response.choices]

    def _stream_response(
        self,
        response: AI21Stream[ChatCompletionChunk],
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in response:
            converted_message = self._convert_ai21_chunk_to_chunk(chunk)
            yield ChatGenerationChunk(message=converted_message)

    def _convert_ai21_chunk_to_chunk(
        self,
        chunk: ChatCompletionChunk,
    ) -> BaseMessageChunk:
        usage = chunk.usage
        content = chunk.choices[0].delta.content or ""

        if usage is None:
            return AIMessageChunk(
                content=content,
            )

        return AIMessageChunk(
            content=content,
            usage_metadata=UsageMetadata(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )
