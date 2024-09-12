"""Test Reka Chat wrapper."""

import os
from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models import ChatReka
from langchain_community.chat_models.reka import (
    convert_to_reka_messages,
    process_content,
)

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
    ("content", "expected"),
    [
        ("Hello", [{"type": "text", "text": "Hello"}]),
        (
            [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": "https://example.com/image.jpg"},
            ],
            [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": "https://example.com/image.jpg"},
            ],
        ),
        (
            [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
            [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": "https://example.com/image.jpg"},
            ],
        ),
    ],
)
def test_process_content(content, expected) -> None:
    result = process_content(content)
    assert result == expected


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [HumanMessage(content="Hello")],
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        ),
        (
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ]
                ),
                AIMessage(content="It's a beautiful landscape."),
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "It's a beautiful landscape."}
                    ],
                },
            ],
        ),
    ],
)
def test_convert_to_reka_messages(
    messages: List[BaseMessage], expected: List[dict]
) -> None:
    result = convert_to_reka_messages(messages)
    assert [message.dict() for message in result] == expected


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
