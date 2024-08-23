from typing import List

import pytest
from ai21.models import ChatMessage as J2ChatMessage
from ai21.models import RoleType
from ai21.models.chat import ChatMessage
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages import (
    ChatMessage as LangChainChatMessage,
)

from langchain_ai21.chat.chat_adapter import ChatAdapter

_J2_MODEL_NAME = "j2-ultra"
_JAMBA_MODEL_NAME = "jamba-instruct-preview"


@pytest.mark.parametrize(
    ids=[
        "when_human_message_j2_model",
        "when_ai_message_j2_model",
        "when_human_message_jamba_model",
        "when_ai_message_jamba_model",
    ],
    argnames=["model", "message", "expected_ai21_message"],
    argvalues=[
        (
            _J2_MODEL_NAME,
            HumanMessage(content="Human Message Content"),
            J2ChatMessage(role=RoleType.USER, text="Human Message Content"),
        ),
        (
            _J2_MODEL_NAME,
            AIMessage(content="AI Message Content"),
            J2ChatMessage(role=RoleType.ASSISTANT, text="AI Message Content"),
        ),
        (
            _JAMBA_MODEL_NAME,
            HumanMessage(content="Human Message Content"),
            ChatMessage(role=RoleType.USER, content="Human Message Content"),
        ),
        (
            _JAMBA_MODEL_NAME,
            AIMessage(content="AI Message Content"),
            ChatMessage(role=RoleType.ASSISTANT, content="AI Message Content"),
        ),
    ],
)
def test_convert_message_to_ai21_message(
    message: BaseMessage,
    expected_ai21_message: ChatMessage,
    chat_adapter: ChatAdapter,
) -> None:
    ai21_message = chat_adapter._convert_message_to_ai21_message(message)
    assert ai21_message == expected_ai21_message


@pytest.mark.parametrize(
    ids=[
        "when_system_message_j2_model",
        "when_langchain_chat_message_j2_model",
    ],
    argnames=["model", "message"],
    argvalues=[
        (
            _J2_MODEL_NAME,
            SystemMessage(content="System Message Content"),
        ),
        (
            _J2_MODEL_NAME,
            LangChainChatMessage(content="Chat Message Content", role="human"),
        ),
    ],
)
def test_convert_message_to_ai21_message__when_invalid_role__should_raise_exception(
    message: BaseMessage,
    chat_adapter: ChatAdapter,
) -> None:
    with pytest.raises(ValueError) as e:
        chat_adapter._convert_message_to_ai21_message(message)
    assert e.value.args[0] == (
        f"Could not resolve role type from message {message}. "
        f"Only support {HumanMessage.__name__} and {AIMessage.__name__}."
    )


@pytest.mark.parametrize(
    ids=[
        "when_all_messages_are_human_messages__should_return_system_none_j2_model",
        "when_first_message_is_system__should_return_system_j2_model",
        "when_all_messages_are_human_messages__should_return_system_none_jamba_model",
        "when_first_message_is_system__should_return_system_jamba_model",
    ],
    argnames=["model", "messages", "expected_messages"],
    argvalues=[
        (
            _J2_MODEL_NAME,
            [
                HumanMessage(content="Human Message Content 1"),
                HumanMessage(content="Human Message Content 2"),
            ],
            {
                "system": "",
                "messages": [
                    J2ChatMessage(
                        role=RoleType.USER,
                        text="Human Message Content 1",
                    ),
                    J2ChatMessage(
                        role=RoleType.USER,
                        text="Human Message Content 2",
                    ),
                ],
            },
        ),
        (
            _J2_MODEL_NAME,
            [
                SystemMessage(content="System Message Content 1"),
                HumanMessage(content="Human Message Content 1"),
            ],
            {
                "system": "System Message Content 1",
                "messages": [
                    J2ChatMessage(
                        role=RoleType.USER,
                        text="Human Message Content 1",
                    ),
                ],
            },
        ),
        (
            _JAMBA_MODEL_NAME,
            [
                HumanMessage(content="Human Message Content 1"),
                HumanMessage(content="Human Message Content 2"),
            ],
            {
                "messages": [
                    ChatMessage(
                        role=RoleType.USER,
                        content="Human Message Content 1",
                    ),
                    ChatMessage(
                        role=RoleType.USER,
                        content="Human Message Content 2",
                    ),
                ]
            },
        ),
        (
            _JAMBA_MODEL_NAME,
            [
                SystemMessage(content="System Message Content 1"),
                HumanMessage(content="Human Message Content 1"),
            ],
            {
                "messages": [
                    ChatMessage(role="system", content="System Message Content 1"),
                    ChatMessage(role="user", content="Human Message Content 1"),
                ],
            },
        ),
    ],
)
def test_convert_messages(
    chat_adapter: ChatAdapter,
    messages: List[BaseMessage],
    expected_messages: List[ChatMessage],
) -> None:
    converted_messages = chat_adapter.convert_messages(messages)
    assert converted_messages == expected_messages


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
    ],
    argnames=["model"],
    argvalues=[
        (_J2_MODEL_NAME,),
    ],
)
def test_convert_messages__when_system_is_not_first(chat_adapter: ChatAdapter) -> None:
    messages = [
        HumanMessage(content="Human Message Content 1"),
        SystemMessage(content="System Message Content 1"),
    ]
    with pytest.raises(ValueError):
        chat_adapter.convert_messages(messages)
