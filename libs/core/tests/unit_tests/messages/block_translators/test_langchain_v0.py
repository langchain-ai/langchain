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


def test_convert_v0_block_with_id() -> None:
    """Test that legacy v0 multimodal blocks carrying an id do not raise TypeError."""
    blocks = [
        {"type": "image", "source_type": "url", "url": "https://example.com/img.png", "id": "block-img-url"},
        {"type": "image", "source_type": "base64", "data": "abc", "mime_type": "image/png", "id": "block-img-b64"},
        {"type": "audio", "source_type": "url", "url": "https://example.com/aud.mp3", "id": "block-aud-url"},
        {"type": "audio", "source_type": "base64", "data": "abc", "mime_type": "audio/mp3", "id": "block-aud-b64"},
        {"type": "file", "source_type": "url", "url": "https://example.com/doc.pdf", "id": "block-file-url"},
        {"type": "file", "source_type": "base64", "data": "abc", "mime_type": "application/pdf", "id": "block-file-b64"},
        {"type": "file", "source_type": "text", "url": "some text content", "id": "block-file-text"},
    ]
    message = HumanMessage(content=blocks)
    converted = message.content_blocks
    assert len(converted) == 7
    for block, original in zip(converted, blocks):
        assert block.get("id") == original["id"]

