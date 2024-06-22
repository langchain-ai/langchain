from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, cast

from ai21.models import ChatMessage as J2ChatMessage
from ai21.models import RoleType
from ai21.models.chat import ChatMessage
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

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

    @abstractmethod
    def call(self, client: Any, **params: Any) -> List[BaseMessage]:
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

    def call(self, client: Any, **params: Any) -> List[BaseMessage]:
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

    def call(self, client: Any, **params: Any) -> List[BaseMessage]:
        response = client.chat.completions.create(**params)

        return [AIMessage(choice.message.content) for choice in response.choices]
