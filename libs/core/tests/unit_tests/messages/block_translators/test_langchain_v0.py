from langchain_core.messages import HumanMessage
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.langchain_v0 import (
    _convert_legacy_v0_content_block_to_v1,
    _get_mime_type,
)
from tests.unit_tests.language_models.chat_models.test_base import (
    _content_blocks_equal_ignore_id,
)


def test_get_mime_type_snake_case() -> None:
    """Test that _get_mime_type handles snake_case mime_type."""
    block = {"mime_type": "image/png"}
    assert _get_mime_type(block) == "image/png"


def test_get_mime_type_camel_case() -> None:
    """Test that _get_mime_type handles camelCase mimeType (from JS SDK)."""
    block = {"mimeType": "image/png"}
    assert _get_mime_type(block) == "image/png"


def test_get_mime_type_both_keys_snake_priority() -> None:
    """Test that _get_mime_type prefers snake_case when both are present."""
    block = {"mime_type": "image/png", "mimeType": "image/jpeg"}
    # snake_case should take priority
    assert _get_mime_type(block) == "image/png"


def test_get_mime_type_none_values() -> None:
    """Test that _get_mime_type handles None values correctly."""
    block = {"mime_type": None, "mimeType": "image/png"}
    # None is falsy, so should fall back to camelCase
    assert _get_mime_type(block) == "image/png"


def test_get_mime_type_empty_string() -> None:
    """Test that _get_mime_type handles empty strings correctly."""
    block = {"mime_type": "", "mimeType": "image/png"}
    # Empty string is falsy, so should fall back to camelCase
    assert _get_mime_type(block) == "image/png"


def test_get_mime_type_missing() -> None:
    """Test that _get_mime_type returns None when key is missing."""
    block = {"url": "https://example.com/image.png"}
    assert _get_mime_type(block) is None


def test_convert_image_url_with_camel_case_mime_type() -> None:
    """Test converting image block with camelCase mimeType (from JS SDK)."""
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/image.png",
        "mimeType": "image/png",  # camelCase from JS SDK
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "image"
    assert result["url"] == "https://example.com/image.png"
    assert result["mime_type"] == "image/png"
    # mimeType should NOT be in extras
    assert "extras" not in result or "mimeType" not in result.get("extras", {})


def test_convert_image_base64_with_camel_case_mime_type() -> None:
    """Test converting base64 image block with camelCase mimeType."""
    block = {
        "type": "image",
        "source_type": "base64",
        "data": "iVBORw0KGgo=",
        "mimeType": "image/png",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "image"
    assert result["base64"] == "iVBORw0KGgo="
    assert result["mime_type"] == "image/png"


def test_convert_audio_url_with_camel_case_mime_type() -> None:
    """Test converting audio URL block with camelCase mimeType."""
    block = {
        "type": "audio",
        "source_type": "url",
        "url": "https://example.com/audio.mp3",
        "mimeType": "audio/mpeg",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "audio"
    assert result["url"] == "https://example.com/audio.mp3"
    assert result["mime_type"] == "audio/mpeg"


def test_convert_audio_base64_with_camel_case_mime_type() -> None:
    """Test converting base64 audio block with camelCase mimeType."""
    block = {
        "type": "audio",
        "source_type": "base64",
        "data": "//uQxAAAA",
        "mimeType": "audio/wav",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "audio"
    assert result["base64"] == "//uQxAAAA"
    assert result["mime_type"] == "audio/wav"


def test_convert_file_url_with_camel_case_mime_type() -> None:
    """Test converting file URL block with camelCase mimeType."""
    block = {
        "type": "file",
        "source_type": "url",
        "url": "https://example.com/document.pdf",
        "mimeType": "application/pdf",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "file"
    assert result["url"] == "https://example.com/document.pdf"
    assert result["mime_type"] == "application/pdf"


def test_convert_file_base64_with_camel_case_mime_type() -> None:
    """Test converting base64 file block with camelCase mimeType."""
    block = {
        "type": "file",
        "source_type": "base64",
        "data": "JVBERi0xLjQ=",
        "mimeType": "application/pdf",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "file"
    assert result["base64"] == "JVBERi0xLjQ="
    assert result["mime_type"] == "application/pdf"


def test_convert_file_text_with_camel_case_mime_type() -> None:
    """Test converting text file block with camelCase mimeType."""
    block = {
        "type": "file",
        "source_type": "text",
        "url": "This is the text content",
        "mimeType": "text/markdown",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "text-plain"
    assert result["text"] == "This is the text content"
    assert result["mime_type"] == "text/markdown"


def test_camel_case_mime_type_not_in_extras() -> None:
    """Test that camelCase mimeType is not included in extras after conversion."""
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/image.png",
        "mimeType": "image/png",
        "alt_text": "An image",  # This should go to extras
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    # alt_text should be in extras
    assert result.get("extras", {}).get("alt_text") == "An image"
    # mimeType should NOT be in extras (it's consumed as mime_type)
    assert "mimeType" not in result.get("extras", {})


def test_convert_message_with_js_sdk_format() -> None:
    """Test full message conversion with JS SDK format (camelCase)."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello from JS SDK"},
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mimeType": "image/jpeg",  # camelCase from JS
            },
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<audio data>",
                "mimeType": "audio/mp3",  # camelCase from JS
            },
            {
                "type": "file",
                "source_type": "base64",
                "data": "<pdf data>",
                "mimeType": "application/pdf",  # camelCase from JS
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {"type": "text", "text": "Hello from JS SDK"},
        {
            "type": "image",
            "base64": "<base64 data>",
            "mime_type": "image/jpeg",
        },
        {
            "type": "audio",
            "base64": "<audio data>",
            "mime_type": "audio/mp3",
        },
        {
            "type": "file",
            "base64": "<pdf data>",
            "mime_type": "application/pdf",
        },
    ]

    assert _content_blocks_equal_ignore_id(message.content_blocks, expected)


def test_backward_compatibility_snake_case() -> None:
    """Test that existing snake_case format still works after the fix."""
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "image/png",  # snake_case (Python format)
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {
            "type": "image",
            "base64": "<base64 data>",
            "mime_type": "image/png",
        },
    ]

    assert _content_blocks_equal_ignore_id(message.content_blocks, expected)


def test_convert_to_v1_from_openai_input() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "image/png",
            },
            {
                "type": "file",
                "source_type": "url",
                "url": "<document url>",
            },
            {
                "type": "file",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "application/pdf",
            },
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "audio/mpeg",
            },
            {
                "type": "file",
                "source_type": "id",
                "id": "<file id>",
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {"type": "text", "text": "Hello"},
        {
            "type": "image",
            "url": "https://example.com/image.png",
        },
        {
            "type": "image",
            "base64": "<base64 data>",
            "mime_type": "image/png",
        },
        {
            "type": "file",
            "url": "<document url>",
        },
        {
            "type": "file",
            "base64": "<base64 data>",
            "mime_type": "application/pdf",
        },
        {
            "type": "audio",
            "base64": "<base64 data>",
            "mime_type": "audio/mpeg",
        },
        {
            "type": "file",
            "file_id": "<file id>",
        },
    ]

    assert _content_blocks_equal_ignore_id(message.content_blocks, expected)


def test_convert_with_extras_on_v0_block() -> None:
    """Test that extras on old-style blocks are preserved in conversion.

    Refer to `_extract_v0_extras` for details.
    """
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/image.png",
        # extras follow
        "alt_text": "An example image",
        "caption": "Example caption",
        "name": "example_image",
        "description": None,
        "attribution": None,
    }
    expected_output = {
        "type": "image",
        "url": "https://example.com/image.png",
        "extras": {
            "alt_text": "An example image",
            "caption": "Example caption",
            "name": "example_image",
            # "description": None,  # These are filtered out
            # "attribution": None,
        },
    }

    assert _convert_legacy_v0_content_block_to_v1(block) == expected_output
