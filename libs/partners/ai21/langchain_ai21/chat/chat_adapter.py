from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union, cast

from ai21.models import ChatMessage as J2ChatMessage
from ai21.models import RoleType
from ai21.models.chat import ChatMessage
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_ChatMessageTypes = Union[ChatMessage, J2ChatMessage]
_ConvertMessagesReturnType = Union[List[_ChatMessageTypes]]
_SYSTEM_ERR_MESSAGE = "System message must be at beginning of message list."


class ChatAdapter(ABC):
    """
    Provides a common interface for the different Chat models available in AI21.
    It converts LangChain messages to AI21 messages.
    Calls the appropriate AI21 model API with the converted messages.
    """

    def convert_messages(
        self,
        messages: List[BaseMessage],
    ) -> Tuple[Optional[str], _ConvertMessagesReturnType]:
        system_message: Optional[str] = None
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

        return system_message, converted_messages

    def _convert_message_to_ai21_message(
        self,
        message: BaseMessage,
    ) -> _ChatMessageTypes:
        content = cast(str, message.content)

        role = None

        if isinstance(message, HumanMessage):
            role = RoleType.USER
        elif isinstance(message, AIMessage):
            role = RoleType.ASSISTANT

        if not role:
            raise ValueError(
                f"Could not resolve role type from message {message}. "
                f"Only support {HumanMessage.__name__} and {AIMessage.__name__}."
            )

        return self._chat_message(role=role, content=content)

    @abstractmethod
    def _chat_message(
        self,
        role: RoleType,
        content: str,
    ) -> _ChatMessageTypes:
        pass

    @abstractmethod
    def call(self, client: Any, **params: Any) -> BaseMessage:
        pass

    def _get_system_message_from_message(self, message: BaseMessage) -> str:
        if not isinstance(message.content, str):
            raise ValueError(
                f"System Message must be of type str. Got {type(message.content)}"
            )

        return message.content


class J2ChatAdapter(ChatAdapter):
    def _chat_message(
        self,
        role: RoleType,
        content: str,
    ) -> J2ChatMessage:
        return J2ChatMessage(role=role, text=content)

    def call(self, client: Any, **params: Any) -> BaseMessage:
        response = client.chat.create(**params)
        outputs = response.outputs
        return AIMessage(content=outputs[0].text)


class JambaChatCompletionsAdapter(ChatAdapter):
    def _chat_message(
        self,
        role: RoleType,
        content: str,
    ) -> ChatMessage:
        return ChatMessage(role=role, content=content)

    def call(self, client: Any, **params: Any) -> BaseMessage:
        response = client.chat.completions.create(**params)
        choices = response.choices
        return AIMessage(content=choices[0].message.content)
