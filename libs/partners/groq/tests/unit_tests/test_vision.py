"""Unit tests for Groq vision support."""

import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import HumanMessage

from langchain_groq.chat_models import ChatGroq

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "fake-key"


def test_vision_with_supported_model() -> None:
    """Vision should work with llama-4-scout."""
    chat = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
                },
            },
        ]
    )

    # Mock the client to avoid actual API calls
    mock_client = MagicMock()
    mock_completion = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is an image of a die.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }
    mock_client.create = MagicMock(return_value=mock_completion)

    with patch.object(chat, "client", mock_client):
        # This should not raise
        response = chat.invoke([message])
        assert len(response.content) > 0
        # Verify the message was converted correctly
        call_args = mock_client.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image_url"


def test_vision_with_unsupported_model() -> None:
    """Vision should fail with non-vision model."""
    chat = ChatGroq(model="llama-3.1-8b-instant")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        ]
    )

    with pytest.raises(ValueError, match="does not support vision"):
        chat.invoke([message])


def test_text_only_still_works() -> None:
    """Text-only messages should still work."""
    chat = ChatGroq(model="llama-3.1-8b-instant")

    # Mock the client to avoid actual API calls
    mock_client = MagicMock()
    mock_completion = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "llama-3.1-8b-instant",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm doing well, thank you.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }
    mock_client.create = MagicMock(return_value=mock_completion)

    message = HumanMessage("Hello, how are you?")

    with patch.object(chat, "client", mock_client):
        response = chat.invoke([message])
        assert len(response.content) > 0
        # Verify text-only messages are passed through correctly
        call_args = mock_client.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how are you?"

