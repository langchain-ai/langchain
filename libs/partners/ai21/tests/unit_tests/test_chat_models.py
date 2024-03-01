"""Test chat model integration."""
from typing import List, Optional
from unittest.mock import Mock, call

import pytest
from ai21 import MissingApiKeyError
from ai21.models import ChatMessage, Penalty, RoleType
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages import (
    ChatMessage as LangChainChatMessage,
)

from langchain_ai21.chat_models import (
    ChatAI21,
    _convert_message_to_ai21_message,
    _convert_messages_to_ai21_messages,
)
from tests.unit_tests.conftest import (
    BASIC_EXAMPLE_LLM_PARAMETERS,
    DUMMY_API_KEY,
    temporarily_unset_api_key,
)


def test_initialization__when_no_api_key__should_raise_exception() -> None:
    """Test integration initialization."""
    with temporarily_unset_api_key():
        with pytest.raises(MissingApiKeyError):
            ChatAI21(model="j2-ultra")


def test_initialization__when_default_parameters_in_init() -> None:
    """Test chat model initialization."""
    ChatAI21(api_key=DUMMY_API_KEY, model="j2-ultra")


def test_initialization__when_custom_parameters_in_init() -> None:
    model = "j2-mid"
    num_results = 1
    max_tokens = 10
    min_tokens = 20
    temperature = 0.1
    top_p = 0.1
    top_k_returns = 0
    frequency_penalty = Penalty(scale=0.2, apply_to_numbers=True)
    presence_penalty = Penalty(scale=0.2, apply_to_stopwords=True)
    count_penalty = Penalty(scale=0.2, apply_to_punctuation=True, apply_to_emojis=True)

    llm = ChatAI21(
        api_key=DUMMY_API_KEY,
        model=model,
        num_results=num_results,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k_returns=top_k_returns,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        count_penalty=count_penalty,
    )
    assert llm.model == model
    assert llm.num_results == num_results
    assert llm.max_tokens == max_tokens
    assert llm.min_tokens == min_tokens
    assert llm.temperature == temperature
    assert llm.top_p == top_p
    assert llm.top_k_return == top_k_returns
    assert llm.frequency_penalty == frequency_penalty
    assert llm.presence_penalty == presence_penalty
    assert count_penalty == count_penalty


@pytest.mark.parametrize(
    ids=[
        "when_human_message",
        "when_ai_message",
    ],
    argnames=["message", "expected_ai21_message"],
    argvalues=[
        (
            HumanMessage(content="Human Message Content"),
            ChatMessage(role=RoleType.USER, text="Human Message Content"),
        ),
        (
            AIMessage(content="AI Message Content"),
            ChatMessage(role=RoleType.ASSISTANT, text="AI Message Content"),
        ),
    ],
)
def test_convert_message_to_ai21_message(
    message: BaseMessage, expected_ai21_message: ChatMessage
) -> None:
    ai21_message = _convert_message_to_ai21_message(message)
    assert ai21_message == expected_ai21_message


@pytest.mark.parametrize(
    ids=[
        "when_system_message",
        "when_langchain_chat_message",
    ],
    argnames=["message"],
    argvalues=[
        (SystemMessage(content="System Message Content"),),
        (LangChainChatMessage(content="Chat Message Content", role="human"),),
    ],
)
def test_convert_message_to_ai21_message__when_invalid_role__should_raise_exception(
    message: BaseMessage,
) -> None:
    with pytest.raises(ValueError) as e:
        _convert_message_to_ai21_message(message)
    assert e.value.args[0] == (
        f"Could not resolve role type from message {message}. "
        f"Only support {HumanMessage.__name__} and {AIMessage.__name__}."
    )


@pytest.mark.parametrize(
    ids=[
        "when_all_messages_are_human_messages__should_return_system_none",
        "when_first_message_is_system__should_return_system",
    ],
    argnames=["messages", "expected_system", "expected_messages"],
    argvalues=[
        (
            [
                HumanMessage(content="Human Message Content 1"),
                HumanMessage(content="Human Message Content 2"),
            ],
            None,
            [
                ChatMessage(role=RoleType.USER, text="Human Message Content 1"),
                ChatMessage(role=RoleType.USER, text="Human Message Content 2"),
            ],
        ),
        (
            [
                SystemMessage(content="System Message Content 1"),
                HumanMessage(content="Human Message Content 1"),
            ],
            "System Message Content 1",
            [
                ChatMessage(role=RoleType.USER, text="Human Message Content 1"),
            ],
        ),
    ],
)
def test_convert_messages(
    messages: List[BaseMessage],
    expected_system: Optional[str],
    expected_messages: List[ChatMessage],
) -> None:
    system, ai21_messages = _convert_messages_to_ai21_messages(messages)
    assert ai21_messages == expected_messages
    assert system == expected_system


def test_convert_messages_when_system_is_not_first__should_raise_value_error() -> None:
    messages = [
        HumanMessage(content="Human Message Content 1"),
        SystemMessage(content="System Message Content 1"),
    ]
    with pytest.raises(ValueError):
        _convert_messages_to_ai21_messages(messages)


def test_invoke(mock_client_with_chat: Mock) -> None:
    chat_input = "I'm Pickle Rick"

    llm = ChatAI21(
        model="j2-ultra",
        api_key=DUMMY_API_KEY,
        client=mock_client_with_chat,
        **BASIC_EXAMPLE_LLM_PARAMETERS,
    )
    llm.invoke(input=chat_input, config=dict(tags=["foo"]))

    mock_client_with_chat.chat.create.assert_called_once_with(
        model="j2-ultra",
        messages=[ChatMessage(role=RoleType.USER, text=chat_input)],
        system="",
        stop_sequences=None,
        **BASIC_EXAMPLE_LLM_PARAMETERS,
    )


def test_generate(mock_client_with_chat: Mock) -> None:
    messages0 = [
        HumanMessage(content="I'm Pickle Rick"),
        AIMessage(content="Hello Pickle Rick! I am your AI Assistant"),
        HumanMessage(content="Nice to meet you."),
    ]
    messages1 = [
        SystemMessage(content="system message"),
        HumanMessage(content="What is 1 + 1"),
    ]
    llm = ChatAI21(
        model="j2-ultra",
        client=mock_client_with_chat,
        **BASIC_EXAMPLE_LLM_PARAMETERS,
    )

    llm.generate(messages=[messages0, messages1])
    mock_client_with_chat.chat.create.assert_has_calls(
        [
            call(
                model="j2-ultra",
                messages=[
                    ChatMessage(
                        role=RoleType.USER,
                        text=str(messages0[0].content),
                    ),
                    ChatMessage(
                        role=RoleType.ASSISTANT, text=str(messages0[1].content)
                    ),
                    ChatMessage(role=RoleType.USER, text=str(messages0[2].content)),
                ],
                system="",
                stop_sequences=None,
                **BASIC_EXAMPLE_LLM_PARAMETERS,
            ),
            call(
                model="j2-ultra",
                messages=[
                    ChatMessage(role=RoleType.USER, text=str(messages1[1].content)),
                ],
                system="system message",
                stop_sequences=None,
                **BASIC_EXAMPLE_LLM_PARAMETERS,
            ),
        ]
    )
