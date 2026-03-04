from langchain_core.messages import HumanMessage
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.langchain_v0 import (
    _convert_legacy_v0_content_block_to_v1,
)
from tests.unit_tests.language_models.chat_models.test_base import (
    _content_blocks_equal_ignore_id,
)


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


def test_convert_camelCase_to_snake_case() -> None:
    """Test that camelCase keys from JS SDK are normalized to snake_case.

    Issue: LangGraph JS SDK generates camelCase (mimeType, sourceType),
    but Python expects snake_case (mime_type, source_type).
    """
    # Test mimeType -> mime_type conversion
    block_image = {
        "type": "image",
        "sourceType": "base64",  # camelCase
        "data": "<base64 data>",
        "mimeType": "image/png",  # camelCase
    }
    result_image = _convert_legacy_v0_content_block_to_v1(block_image)
    assert result_image["type"] == "image"
    assert result_image["base64"] == "<base64 data>"
    assert result_image["mime_type"] == "image/png"

    # Test audio with camelCase
    block_audio = {
        "type": "audio",
        "sourceType": "url",  # camelCase
        "url": "https://example.com/audio.mp3",
        "mimeType": "audio/mpeg",  # camelCase
    }
    result_audio = _convert_legacy_v0_content_block_to_v1(block_audio)
    assert result_audio["type"] == "audio"
    assert result_audio["url"] == "https://example.com/audio.mp3"
    assert result_audio["mime_type"] == "audio/mpeg"

    # Test file with camelCase
    block_file = {
        "type": "file",
        "sourceType": "base64",  # camelCase
        "data": "<base64 data>",
        "mimeType": "application/pdf",  # camelCase
    }
    result_file = _convert_legacy_v0_content_block_to_v1(block_file)
    assert result_file["type"] == "file"
    assert result_file["base64"] == "<base64 data>"
    assert result_file["mime_type"] == "application/pdf"


def test_convert_snake_case_unchanged() -> None:
    """Test that existing snake_case blocks still work correctly."""
    block = {
        "type": "image",
        "source_type": "base64",
        "data": "<base64 data>",
        "mime_type": "image/png",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "image"
    assert result["base64"] == "<base64 data>"
    assert result["mime_type"] == "image/png"
