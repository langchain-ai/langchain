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


def test_convert_to_v1_from_camel_case_input() -> None:
    """Test that JS-style camelCase v0 blocks are normalized and converted."""
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "sourceType": "url",
                "url": "https://example.com/image.png",
                "mimeType": "image/png",
            },
            {
                "type": "audio",
                "sourceType": "base64",
                "data": "<audio base64 data>",
                "mimeType": "audio/mpeg",
            },
            {
                "type": "file",
                "sourceType": "base64",
                "data": "<file base64 data>",
                "mimeType": "application/pdf",
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {
            "type": "image",
            "url": "https://example.com/image.png",
            "mime_type": "image/png",
        },
        {
            "type": "audio",
            "base64": "<audio base64 data>",
            "mime_type": "audio/mpeg",
        },
        {
            "type": "file",
            "base64": "<file base64 data>",
            "mime_type": "application/pdf",
        },
    ]

    assert _content_blocks_equal_ignore_id(message.content_blocks, expected)


def test_convert_legacy_block_prefers_snake_case_when_both_key_styles_present() -> None:
    """Test that snake_case wins if both snake_case and camelCase are provided."""
    block = {
        "type": "image",
        "source_type": "url",
        "sourceType": "base64",
        "url": "https://example.com/image.png",
        "mime_type": "image/snake",
        "mimeType": "image/camel",
    }

    expected_output = {
        "type": "image",
        "url": "https://example.com/image.png",
        "mime_type": "image/snake",
    }

    assert _convert_legacy_v0_content_block_to_v1(block) == expected_output
    assert block["sourceType"] == "base64"
    assert block["mimeType"] == "image/camel"
