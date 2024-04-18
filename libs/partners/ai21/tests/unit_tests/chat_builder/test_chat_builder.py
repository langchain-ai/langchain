from typing import List, Optional

import pytest
from ai21.models import RoleType

from langchain_ai21.chat_builder.chat_builder import ChatBuilder
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from ai21.models.chat import ChatMessage
from ai21.models import ChatMessage as J2ChatMessage
from langchain_core.messages import (
    ChatMessage as LangChainChatMessage,
)
from langchain_ai21.chat_builder.chat_builder_factory import create_chat_builder

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
        chat_builder: ChatBuilder
) -> None:
    ai21_message = chat_builder._convert_message_to_ai21_message(message)
    assert ai21_message == expected_ai21_message


@pytest.mark.parametrize(
    ids=[
        "when_system_message_j2_model",
        "when_langchain_chat_message_j2_model",
        "when_system_message_jamba_model",
        "when_langchain_chat_message_jamba_model",
    ],
    argnames=["model", "message"],
    argvalues=[
        (_J2_MODEL_NAME, SystemMessage(content="System Message Content")),
        (_J2_MODEL_NAME, LangChainChatMessage(content="Chat Message Content", role="human"),),
        (_JAMBA_MODEL_NAME, SystemMessage(content="System Message Content")),
        (_JAMBA_MODEL_NAME, LangChainChatMessage(content="Chat Message Content", role="human"),),
    ],
)
def test_convert_message_to_ai21_message__when_invalid_role__should_raise_exception(
        message: BaseMessage, chat_builder: ChatBuilder,
) -> None:
    with pytest.raises(ValueError) as e:
        chat_builder._convert_message_to_ai21_message(message)
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
    argnames=["model", "messages", "expected_system", "expected_messages"],
    argvalues=[
        (
                _J2_MODEL_NAME,
                [
                    HumanMessage(content="Human Message Content 1"),
                    HumanMessage(content="Human Message Content 2"),
                ],
                None,
                [
                    J2ChatMessage(role=RoleType.USER, text="Human Message Content 1"),
                    J2ChatMessage(role=RoleType.USER, text="Human Message Content 2"),
                ],
        ),
        (
                _J2_MODEL_NAME,
                [
                    SystemMessage(content="System Message Content 1"),
                    HumanMessage(content="Human Message Content 1"),
                ],
                "System Message Content 1",
                [
                    J2ChatMessage(role=RoleType.USER, text="Human Message Content 1"),
                ],
        ),
        (
                _JAMBA_MODEL_NAME,
                [
                    HumanMessage(content="Human Message Content 1"),
                    HumanMessage(content="Human Message Content 2"),
                ],
                None,
                [
                    ChatMessage(role=RoleType.USER, content="Human Message Content 1"),
                    ChatMessage(role=RoleType.USER, content="Human Message Content 2"),
                ],
        ),
        (
                _JAMBA_MODEL_NAME,
                [
                    SystemMessage(content="System Message Content 1"),
                    HumanMessage(content="Human Message Content 1"),
                ],
                "System Message Content 1",
                [
                    ChatMessage(role=RoleType.USER, content="Human Message Content 1"),
                ],
        ),
    ],
)
def test_build(
        chat_builder: ChatBuilder,
        messages: List[BaseMessage],
        expected_system: Optional[str],
        expected_messages: List[ChatMessage],
) -> None:
    system, ai21_messages = chat_builder.build(messages)
    assert ai21_messages == expected_messages
    assert system == expected_system


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
    ],
    argnames=["model"],
    argvalues=[
        (_J2_MODEL_NAME,),
        (_JAMBA_MODEL_NAME,),
    ],
)
def test_build__when_system_is_not_first__should_raise_value_error(chat_builder: ChatBuilder) -> None:
    messages = [
        HumanMessage(content="Human Message Content 1"),
        SystemMessage(content="System Message Content 1"),
    ]
    with pytest.raises(ValueError):
        chat_builder.build(messages)
