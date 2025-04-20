import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from langchain_community.chat_models import ChatReka
from langchain_community.chat_models.reka import (
    convert_to_reka_messages,
    process_content,
)

os.environ["REKA_API_KEY"] = "dummy_key"


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_model_param() -> None:
    llm = ChatReka(model="reka-flash")
    assert llm.model == "reka-flash"


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_model_kwargs() -> None:
    llm = ChatReka(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_incorrect_field() -> None:
    """Test that providing an incorrect field raises ValidationError."""
    with pytest.raises(ValidationError):
        ChatReka(unknown_field="bar")  # type: ignore[call-arg]


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_initialization() -> None:
    """Test Reka initialization."""
    # Verify that ChatReka can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatReka(model="reka-flash", reka_api_key="test_key")


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
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


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
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


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_streaming() -> None:
    llm = ChatReka(streaming=True)
    assert llm.streaming is True


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_temperature() -> None:
    llm = ChatReka(temperature=0.5)
    assert llm.temperature == 0.5


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_max_tokens() -> None:
    llm = ChatReka(max_tokens=100)
    assert llm.max_tokens == 100


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_default_params() -> None:
    llm = ChatReka()
    assert llm._default_params == {
        "max_tokens": 256,
        "model": "reka-flash",
    }


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_identifying_params() -> None:
    """Test that ChatReka identifies its default parameters correctly."""
    chat = ChatReka(model="reka-flash", temperature=0.7, max_tokens=256)
    expected_params = {
        "model": "reka-flash",
        "temperature": 0.7,
        "max_tokens": 256,
    }
    assert chat._default_params == expected_params


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_llm_type() -> None:
    llm = ChatReka()
    assert llm._llm_type == "reka-chat"


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_tool_use_with_mocked_response() -> None:
    with patch("reka.client.Reka") as MockReka:
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


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        # Test single system message
        (
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Hello"),
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant.\nHello"}
                    ],
                }
            ],
        ),
        # Test system message with multiple messages
        (
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
                HumanMessage(content="Thanks!"),
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant.\nWhat is 2+2?",
                        }
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "4"}]},
                {"role": "user", "content": [{"type": "text", "text": "Thanks!"}]},
            ],
        ),
        # Test system message with media content
        (
            [
                SystemMessage(content="Hi."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ]
                ),
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hi.\nWhat's in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                },
            ],
        ),
    ],
)
def test_system_message_handling(
    messages: List[BaseMessage], expected: List[Dict[str, Any]]
) -> None:
    """Test that system messages are handled correctly."""
    result = convert_to_reka_messages(messages)
    assert result == expected


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_multiple_system_messages_error() -> None:
    """Test that multiple system messages raise an error."""
    messages = [
        SystemMessage(content="System message 1"),
        SystemMessage(content="System message 2"),
        HumanMessage(content="Hello"),
    ]

    with pytest.raises(ValueError, match="Multiple system messages are not supported."):
        convert_to_reka_messages(messages)


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_get_num_tokens() -> None:
    """Test that token counting works correctly for different input types."""
    llm = ChatReka()
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    # Test string input
    text = "What is the weather like today?"
    expected_tokens = len(encoding.encode(text))
    assert llm.get_num_tokens(text) == expected_tokens

    # Test BaseMessage input
    message = HumanMessage(content="What is the weather like today?")
    assert isinstance(message.content, str)
    expected_tokens = len(encoding.encode(message.content))
    assert llm.get_num_tokens(message) == expected_tokens

    # Test List[BaseMessage] input
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi!"),
        AIMessage(content="Hello! How can I help you today?"),
    ]
    expected_tokens = sum(
        len(encoding.encode(msg.content))
        for msg in messages
        if isinstance(msg.content, str)
    )
    assert llm.get_num_tokens(messages) == expected_tokens

    # Test empty message list
    assert llm.get_num_tokens([]) == 0
