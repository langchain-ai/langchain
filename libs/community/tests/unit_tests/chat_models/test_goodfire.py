"""Test Goodfire Chat API wrapper."""

import os
from typing import Any, List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models import ChatGoodfire
from langchain_community.chat_models.goodfire import (
    format_for_goodfire,
    format_for_langchain,
)

os.environ["GOODFIRE_API_KEY"] = "test_key"


def get_valid_variant() -> Any:
    try:
        import goodfire
    except ImportError as e:
        raise ImportError(
            "Could not import goodfire python package. "
            "Please install it with `pip install goodfire`."
        ) from e
    return goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")


@pytest.mark.requires("goodfire")
def test_goodfire_model_param() -> None:
    try:
        import goodfire
    except ImportError as e:
        raise ImportError(
            "Could not import goodfire python package. "
            "Please install it with `pip install goodfire`."
        ) from e
    base_variant = get_valid_variant()
    llm = ChatGoodfire(model=base_variant)
    assert isinstance(llm.model, goodfire.Variant)
    assert llm.model.base_model == base_variant.base_model


@pytest.mark.requires("goodfire")
def test_goodfire_initialization() -> None:
    """Test goodfire initialization with API key."""
    try:
        import goodfire
    except ImportError as e:
        raise ImportError(
            "Could not import goodfire python package. "
            "Please install it with `pip install goodfire`."
        ) from e
    llm = ChatGoodfire(model=get_valid_variant(), goodfire_api_key="test_key")
    assert llm.goodfire_api_key.get_secret_value() == "test_key"
    assert isinstance(llm.sync_client, goodfire.Client)
    assert isinstance(llm.async_client, goodfire.AsyncClient)


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
                {"role": "system", "content": "You're an assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ),
    ],
)
def test_message_formatting(messages: List[BaseMessage], expected: List[dict]) -> None:
    result = format_for_goodfire(messages)
    assert result == expected


def test_format_for_langchain() -> None:
    message = {"role": "assistant", "content": "Hello there!"}
    result = format_for_langchain(message)
    assert isinstance(result, AIMessage)
    assert result.content == "Hello there!"


def test_format_for_langchain_invalid_role() -> None:
    message = {"role": "user", "content": "Hello"}
    with pytest.raises(AssertionError, match="Expected role 'assistant'"):
        format_for_langchain(message)


@pytest.mark.requires("goodfire")
def test_invalid_message_type() -> None:
    class CustomMessage(BaseMessage):
        content: str
        type: str = "custom"

    with pytest.raises(ValueError, match="Unknown message type"):
        format_for_goodfire([CustomMessage(content="test")])
