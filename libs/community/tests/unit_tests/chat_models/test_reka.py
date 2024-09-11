"""Test Reka Chat API wrapper."""

import os
from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models import ChatReka
from langchain_community.chat_models.reka import process_messages_for_reka

os.environ["REKA_API_KEY"] = "dummy_key"


@pytest.mark.requires("reka")
def test_reka_model_name_param() -> None:
    llm = ChatReka(model_name="reka-flash")
    assert llm.model == "reka-flash"


@pytest.mark.requires("reka")
def test_reka_model_param() -> None:
    llm = ChatReka(model="reka-flash")
    assert llm.model == "reka-flash"


@pytest.mark.requires("reka")
def test_reka_model_kwargs() -> None:
    llm = ChatReka(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("reka")
def test_reka_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        ChatReka(model_kwargs={"max_tokens": "invalid"})


@pytest.mark.requires("reka")
def test_reka_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatReka(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("reka")
def test_reka_initialization() -> None:
    """Test Reka initialization."""
    # Verify that ChatReka can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatReka(model="reka-flash", reka_api_key="test_key")


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([HumanMessage(content="Hello")], [{"role": "user", "content": "Hello"}]),
        (
            [HumanMessage(content="Hello"), AIMessage(content="Hi there!")],
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ),
        (
            [
                SystemMessage(content="You're an assistant"),
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ],
            [
                {"role": "user", "content": "You're an assistant\nHello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ),
    ],
)
def test_message_processing(messages: List[BaseMessage], expected: List[dict]) -> None:
    result = process_messages_for_reka(messages)
    assert result == expected


@pytest.mark.requires("reka")
def test_reka_streaming() -> None:
    llm = ChatReka(streaming=True)
    assert llm.streaming is True


@pytest.mark.requires("reka")
def test_reka_temperature() -> None:
    llm = ChatReka(temperature=0.5)
    assert llm.temperature == 0.5


@pytest.mark.requires("reka")
def test_reka_max_tokens() -> None:
    llm = ChatReka(max_tokens=100)
    assert llm.max_tokens == 100


@pytest.mark.requires("reka")
def test_reka_default_params() -> None:
    llm = ChatReka()
    assert llm._default_params == {
        "max_tokens": 256,
        "model": "reka-flash",
    }


@pytest.mark.requires("reka")
def test_reka_identifying_params() -> None:
    llm = ChatReka(temperature=0.7)
    assert llm._identifying_params == {
        "max_tokens": 256,
        "model": "reka-flash",
        "temperature": 0.7,
    }


@pytest.mark.requires("reka")
def test_reka_llm_type() -> None:
    llm = ChatReka()
    assert llm._llm_type == "reka-chat"
