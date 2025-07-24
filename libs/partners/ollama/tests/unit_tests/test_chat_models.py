"""Test chat model integration."""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import Client, Request, Response
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ChatMessage, HumanMessage
from langchain_core.messages.content_blocks import (
    AudioContentBlock,
    FileContentBlock,
    ImageContentBlock,
    PlainTextContentBlock,
    TextContentBlock,
    VideoContentBlock,
)
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_ollama.chat_models import (
    ChatOllama,
    _parse_arguments_from_tool_call,
    _parse_json_string,
)

MODEL_NAME = "llama3.1"


class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}


# Tests for new standard content block handling


def test_convert_standard_content_block_to_ollama_text_block() -> None:
    """Test conversion of TextContentBlock to Ollama format."""
    text_block: TextContentBlock = {
        "type": "text",
        "text": "Hello, world!",
    }
    
    text_content, images = _convert_standard_content_block_to_ollama(text_block)
    
    assert text_content == "Hello, world!"
    assert images == []


def test_convert_standard_content_block_to_ollama_image_block_base64() -> None:
    """Test conversion of ImageContentBlock with base64 data to Ollama format."""
    image_block: ImageContentBlock = {
        "type": "image",
        "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "mime_type": "image/png",
    }
    
    text_content, images = _convert_standard_content_block_to_ollama(image_block)
    
    assert text_content == ""
    assert len(images) == 1
    assert images[0] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


def test_convert_standard_content_block_to_ollama_image_block_url_error() -> None:
    """Test that ImageContentBlock with URL raises appropriate error."""
    image_block: ImageContentBlock = {
        "type": "image",
        "url": "https://example.com/image.png",
        "mime_type": "image/png",
    }
    
    with pytest.raises(ValueError, match="Image URLs are not supported. Only base64 image data is supported."):
        _convert_standard_content_block_to_ollama(image_block)


def test_convert_standard_content_block_to_ollama_image_block_missing_data_error() -> None:
    """Test that ImageContentBlock without base64 or url raises appropriate error."""
    image_block: ImageContentBlock = {
        "type": "image",
        "mime_type": "image/png",
    }
    
    with pytest.raises(ValueError, match="ImageContentBlock must contain either 'base64' or 'url' field."):
        _convert_standard_content_block_to_ollama(image_block)


def test_convert_standard_content_block_to_ollama_unsupported_audio_block() -> None:
    """Test that AudioContentBlock raises appropriate error."""
    audio_block: AudioContentBlock = {
        "type": "audio",
        "base64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT",
        "mime_type": "audio/wav",
    }
    
    with pytest.raises(ValueError, match="Content block type 'audio' is not supported by Ollama. Supported types: text, image \\(base64 only\\)."):
        _convert_standard_content_block_to_ollama(audio_block)


def test_convert_standard_content_block_to_ollama_unsupported_video_block() -> None:
    """Test that VideoContentBlock raises appropriate error."""
    video_block: VideoContentBlock = {
        "type": "video",
        "base64": "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAACKBtZGF0AAAC",
        "mime_type": "video/mp4",
    }
    
    with pytest.raises(ValueError, match="Content block type 'video' is not supported by Ollama. Supported types: text, image \\(base64 only\\)."):
        _convert_standard_content_block_to_ollama(video_block)


def test_convert_standard_content_block_to_ollama_unsupported_plaintext_block() -> None:
    """Test that PlainTextContentBlock raises appropriate error."""
    plaintext_block: PlainTextContentBlock = {
        "type": "text-plain",
        "text": "This is plain text content",
    }
    
    with pytest.raises(ValueError, match="Content block type 'text-plain' is not supported by Ollama. Supported types: text, image \\(base64 only\\)."):
        _convert_standard_content_block_to_ollama(plaintext_block)


def test_convert_standard_content_block_to_ollama_unsupported_file_block() -> None:
    """Test that FileContentBlock raises appropriate error."""
    file_block: FileContentBlock = {
        "type": "file",
        "base64": "JVBERi0xLjQKJcOkw7zDtsO8CjIgMCBvYmoKPDwKL0xlbmd0aCAzIDAgUgo+PgpzdHJlYW0K",
        "mime_type": "application/pdf",
        "filename": "document.pdf",
    }
    
    with pytest.raises(ValueError, match="Content block type 'file' is not supported by Ollama. Supported types: text, image \\(base64 only\\)."):
        _convert_standard_content_block_to_ollama(file_block)


def test_get_image_from_data_content_block_legacy_format() -> None:
    """Test _get_image_from_data_content_block with legacy data content block format."""
    legacy_block = {
        "type": "image",
        "source_type": "base64",
        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
    }
    
    result = _get_image_from_data_content_block(legacy_block)
    
    assert result == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


def test_get_image_from_data_content_block_new_base64_format() -> None:
    """Test _get_image_from_data_content_block with new ImageContentBlock base64 format."""
    new_block = {
        "type": "image",
        "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "mime_type": "image/png",
    }
    
    result = _get_image_from_data_content_block(new_block)
    
    assert result == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


def test_get_image_from_data_content_block_new_url_format_error() -> None:
    """Test _get_image_from_data_content_block with new ImageContentBlock URL format raises error."""
    new_block = {
        "type": "image",
        "url": "https://example.com/image.png",
        "mime_type": "image/png",
    }
    
    with pytest.raises(ValueError, match="Image URLs are not supported. Only base64 image data is supported."):
        _get_image_from_data_content_block(new_block)


def test_get_image_from_data_content_block_missing_data_error() -> None:
    """Test _get_image_from_data_content_block with missing data raises appropriate error."""
    incomplete_block = {
        "type": "image",
        "mime_type": "image/png",
    }
    
    with pytest.raises(ValueError, match="Image data only supported through base64 format. Block must contain 'base64' field or legacy 'source_type'/'data' fields."):
        _get_image_from_data_content_block(incomplete_block)


def test_get_image_from_data_content_block_unsupported_type_error() -> None:
    """Test _get_image_from_data_content_block with unsupported block type raises error."""
    unsupported_block = {
        "type": "audio",
        "base64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT",
    }
    
    with pytest.raises(ValueError, match="Blocks of type audio not supported."):
        _get_image_from_data_content_block(unsupported_block)


def test__parse_arguments_from_tool_call() -> None:
    raw_response = '{"model":"sample-model","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_profile_details","arguments":{"arg_1":"12345678901234567890123456"}}}]},"done":false}'  # noqa: E501
    raw_tool_calls = json.loads(raw_response)["message"]["tool_calls"]
    response = _parse_arguments_from_tool_call(raw_tool_calls[0])
    assert response is not None
    assert isinstance(response["arg_1"], str)


@contextmanager
def _mock_httpx_client_stream(
    *args: Any, **kwargs: Any
) -> Generator[Response, Any, Any]:
    yield Response(
        status_code=200,
        content='{"message": {"role": "assistant", "content": "The meaning ..."}}',
        request=Request(method="POST", url="http://whocares:11434"),
    )


def test_arbitrary_roles_accepted_in_chatmessages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Client, "stream", _mock_httpx_client_stream)
    llm = ChatOllama(
        model=MODEL_NAME,
        verbose=True,
        format=None,
    )
    messages = [
        ChatMessage(
            role="somerandomrole",
            content="I'm ok with you adding any role message now!",
        ),
        ChatMessage(role="control", content="thinking"),
        ChatMessage(role="user", content="What is the meaning of life?"),
    ]
    llm.invoke(messages)


@patch("langchain_ollama.chat_models.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    # Test that validate_model is called when validate_model_on_init=True
    ChatOllama(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    # Test that validate_model is NOT called when validate_model_on_init=False
    ChatOllama(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()

    # Test that validate_model is NOT called by default
    ChatOllama(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


# Define a dummy raw_tool_call for the function signature
dummy_raw_tool_call = {
    "function": {"name": "test_func", "arguments": ""},
}


# --- Regression tests for tool-call argument parsing (see #30910) ---


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        # Case 1: Standard double-quoted JSON
        ('{"key": "value", "number": 123}', {"key": "value", "number": 123}),
        # Case 2: Single-quoted string (the original bug)
        ("{'key': 'value', 'number': 123}", {"key": "value", "number": 123}),
        # Case 3: String with an internal apostrophe
        ('{"text": "It\'s a great test!"}', {"text": "It's a great test!"}),
        # Case 4: Mixed quotes that ast can handle
        ("{'text': \"It's a great test!\"}", {"text": "It's a great test!"}),
    ],
)
def test_parse_json_string_success_cases(
    input_string: str, expected_output: Any
) -> None:
    """Tests that _parse_json_string correctly parses valid and fixable strings."""
    raw_tool_call = {"function": {"name": "test_func", "arguments": input_string}}
    result = _parse_json_string(input_string, raw_tool_call=raw_tool_call, skip=False)
    assert result == expected_output


def test_parse_json_string_failure_case_raises_exception() -> None:
    """Tests that _parse_json_string raises an exception for truly malformed strings."""
    malformed_string = "{'key': 'value',,}"
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    with pytest.raises(OutputParserException):
        _parse_json_string(
            malformed_string,
            raw_tool_call=raw_tool_call,
            skip=False,
        )


def test_parse_json_string_skip_returns_input_on_failure() -> None:
    """Tests that skip=True returns the original string on parse failure."""
    malformed_string = "{'not': valid,,,}"
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    result = _parse_json_string(
        malformed_string,
        raw_tool_call=raw_tool_call,
        skip=True,
    )
    assert result == malformed_string


def test_load_response_with_empty_content_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that load responses with empty content log a warning and are skipped."""
    load_only_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": ""},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_only_response

        llm = ChatOllama(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No data received from Ollama stream"),
        ):
            llm.invoke([HumanMessage("Hello")])

        assert "Ollama returned empty response with done_reason='load'" in caplog.text


def test_load_response_with_whitespace_content_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses w/ only whitespace content log a warning and are skipped."""
    load_whitespace_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": "   \n  \t  "},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_whitespace_response

        llm = ChatOllama(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No data received from Ollama stream"),
        ):
            llm.invoke([HumanMessage("Hello")])
        assert "Ollama returned empty response with done_reason='load'" in caplog.text


def test_load_followed_by_content_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses log a warning and are skipped when followed by content."""
    load_then_content_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": ""},
        },
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:01.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
        },
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_then_content_response

        llm = ChatOllama(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessage("Hello")])

        assert "Ollama returned empty response with done_reason='load'" in caplog.text
        assert result.content == "Hello! How can I help you today?"
        assert result.response_metadata.get("done_reason") == "stop"


def test_load_response_with_actual_content_is_not_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses with actual content are NOT skipped and log no warning."""
    load_with_content_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": "This is actual content"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_with_content_response

        llm = ChatOllama(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessage("Hello")])

        assert result.content == "This is actual content"
        assert result.response_metadata.get("done_reason") == "load"
        assert not caplog.text

