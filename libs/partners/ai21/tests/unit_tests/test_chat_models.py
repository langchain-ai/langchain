"""Test chat model integration."""

from typing import cast
from unittest.mock import Mock, call

import pytest
from ai21 import MissingApiKeyError
from ai21.models import ChatMessage, Penalty, RoleType
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_ai21.chat_models import (
    ChatAI21,
)
from tests.unit_tests.conftest import (
    BASIC_EXAMPLE_CHAT_PARAMETERS,
    BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT,
    DUMMY_API_KEY,
    temporarily_unset_api_key,
)


def test_initialization__when_no_api_key__should_raise_exception() -> None:
    """Test integration initialization."""
    with temporarily_unset_api_key():
        with pytest.raises(MissingApiKeyError):
            ChatAI21(model="j2-ultra")  # type: ignore[call-arg]


def test_initialization__when_default_parameters_in_init() -> None:
    """Test chat model initialization."""
    ChatAI21(api_key=DUMMY_API_KEY, model="j2-ultra")  # type: ignore[call-arg, arg-type]


def test_initialization__when_custom_parameters_in_init() -> None:
    model = "j2-ultra"
    num_results = 1
    max_tokens = 10
    min_tokens = 20
    temperature = 0.1
    top_p = 0.1
    top_k_return = 0
    frequency_penalty = Penalty(scale=0.2, apply_to_numbers=True)  # type: ignore[call-arg]
    presence_penalty = Penalty(scale=0.2, apply_to_stopwords=True)  # type: ignore[call-arg]
    count_penalty = Penalty(scale=0.2, apply_to_punctuation=True, apply_to_emojis=True)  # type: ignore[call-arg]

    llm = ChatAI21(  # type: ignore[call-arg]
        api_key=DUMMY_API_KEY,  # type: ignore[arg-type]
        model=model,
        num_results=num_results,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k_return=top_k_return,
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
    assert llm.top_k_return == top_k_return
    assert llm.frequency_penalty == frequency_penalty
    assert llm.presence_penalty == presence_penalty
    assert count_penalty == count_penalty


def test_invoke(mock_client_with_chat: Mock) -> None:
    chat_input = "I'm Pickle Rick"

    llm = ChatAI21(
        model="j2-ultra",
        api_key=DUMMY_API_KEY,  # type: ignore[arg-type]
        client=mock_client_with_chat,
        **BASIC_EXAMPLE_CHAT_PARAMETERS,  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
    )
    llm.invoke(input=chat_input, config=dict(tags=["foo"]), stop=["\n"])

    mock_client_with_chat.chat.create.assert_called_once_with(
        model="j2-ultra",
        messages=[ChatMessage(role=RoleType.USER, text=chat_input)],
        system="",
        stop_sequences=["\n"],
        **BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT,
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
        **BASIC_EXAMPLE_CHAT_PARAMETERS,  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
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
                **BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT,
            ),
            call(
                model="j2-ultra",
                messages=[
                    ChatMessage(role=RoleType.USER, text=str(messages1[1].content)),
                ],
                system="system message",
                **BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT,
            ),
        ]
    )


def test_api_key_is_secret_string() -> None:
    llm = ChatAI21(model="j2-ultra", api_key="secret-api-key")  # type: ignore[call-arg, arg-type]
    assert isinstance(llm.api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("AI21_API_KEY", "secret-api-key")
    llm = ChatAI21(model="j2-ultra")  # type: ignore[call-arg]
    print(llm.api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = ChatAI21(model="j2-ultra", api_key="secret-api-key")  # type: ignore[call-arg, arg-type]
    print(llm.api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    llm = ChatAI21(model="j2-ultra", api_key="secret-api-key")  # type: ignore[call-arg, arg-type]
    assert cast(SecretStr, llm.api_key).get_secret_value() == "secret-api-key"
