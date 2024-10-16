import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from langchain_community.chat_models import ChatReka
from langchain_community.chat_models.reka import (
    convert_to_reka_messages,
    process_content,
)

os.environ["REKA_API_KEY"] = "dummy_key"


@pytest.mark.requires("reka-api")
def test_reka_model_param() -> None:
    llm = ChatReka(model="reka-flash")
    assert llm.model == "reka-flash"


@pytest.mark.requires("reka-api")
def test_reka_model_kwargs() -> None:
    llm = ChatReka(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("reka-api")
def test_reka_incorrect_field() -> None:
    """Test that providing an incorrect field raises ValidationError."""
    with pytest.raises(ValidationError):
        ChatReka(unknown_field="bar")  # type: ignore


@pytest.mark.requires("reka-api")
def test_reka_initialization() -> None:
    """Test Reka initialization."""
    # Verify that ChatReka can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatReka(model="reka-flash", reka_api_key="test_key")


@pytest.mark.requires("reka-api")
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Hello", [{"type": "text", "text": "Hello"}]),
        (
            [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.jpg",
                },
            ],
            [
                {"type": "text", "text": "Describe this image"},
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
def test_process_content(content: Any, expected: List[Dict[str, Any]]) -> None:
    result = process_content(content)
    assert result == expected


@pytest.mark.requires("reka-api")
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
    messages: List[BaseMessage], expected: List[Dict[str, Any]]
) -> None:
    result = convert_to_reka_messages(messages)
    assert result == expected


@pytest.mark.requires("reka-api")
def test_reka_streaming() -> None:
    llm = ChatReka(streaming=True)
    assert llm.streaming is True


@pytest.mark.requires("reka-api")
def test_reka_temperature() -> None:
    llm = ChatReka(temperature=0.5)
    assert llm.temperature == 0.5


@pytest.mark.requires("reka-api")
def test_reka_max_tokens() -> None:
    llm = ChatReka(max_tokens=100)
    assert llm.max_tokens == 100


@pytest.mark.requires("reka-api")
def test_reka_default_params() -> None:
    llm = ChatReka()
    assert llm._default_params == {
        "max_tokens": 256,
        "model": "reka-flash",
    }


@pytest.mark.requires("reka-api")
def test_reka_identifying_params() -> None:
    """Test that ChatReka identifies its default parameters correctly."""
    chat = ChatReka(model="reka-flash", temperature=0.7, max_tokens=256)
    expected_params = {
        "model": "reka-flash",
        "temperature": 0.7,
        "max_tokens": 256,
    }
    assert chat._default_params == expected_params


@pytest.mark.requires("reka-api")
def test_reka_llm_type() -> None:
    llm = ChatReka()
    assert llm._llm_type == "reka-chat"


@pytest.mark.requires("reka-api")
def test_reka_tool_use_with_mocked_response() -> None:
    with patch("langchain_community.chat_models.reka.Reka") as MockReka:
        # Mock the Reka client
        mock_client = MockReka.return_value
        mock_chat = MagicMock()
        mock_client.chat = mock_chat
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "tool_call_1"
        mock_tool_call.name = "search_tool"
        mock_tool_call.parameters = {"query": "LangChain"}
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = None
        mock_response.responses = [MagicMock(message=mock_message)]
        mock_chat.create.return_value = mock_response

        llm = ChatReka()
        messages: List[BaseMessage] = [HumanMessage(content="Tell me about LangChain")]
        result = llm._generate(messages)

        assert len(result.generations) == 1
        ai_message = result.generations[0].message
        assert ai_message.content == ""
        assert "tool_calls" in ai_message.additional_kwargs
        tool_calls = ai_message.additional_kwargs["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "tool_call_1"
        assert tool_calls[0]["function"]["name"] == "search_tool"
        assert tool_calls[0]["function"]["arguments"] == json.dumps(
            {"query": "LangChain"}
        )
