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


def test_v0_block_with_id_does_not_raise_type_error() -> None:
    """Regression test for https://github.com/langchain-ai/langchain/issues/39797.

    A v0 image block that also carries a block `id` should not raise `TypeError`
    when converted via `HumanMessage.content_blocks`.
    """
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/x.png",
                "id": "block-1",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0].get("id") == "block-1"
    assert blocks[0].get("url") == "https://example.com/x.png"


def test_v0_image_base64_with_id() -> None:
    """image/base64 v0 block with id should convert cleanly."""
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "image/png",
                "id": "block-2",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0].get("id") == "block-2"
    assert blocks[0].get("base64") == "<base64 data>"
    assert blocks[0].get("mime_type") == "image/png"


def test_v0_audio_url_with_id() -> None:
    """audio/url v0 block with id should convert cleanly."""
    message = HumanMessage(
        content=[
            {
                "type": "audio",
                "source_type": "url",
                "url": "https://example.com/audio.mp3",
                "id": "audio-1",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "audio"
    assert blocks[0].get("id") == "audio-1"
    assert blocks[0].get("url") == "https://example.com/audio.mp3"


def test_v0_audio_base64_with_id() -> None:
    """audio/base64 v0 block with id should convert cleanly."""
    message = HumanMessage(
        content=[
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "audio/mpeg",
                "id": "audio-2",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "audio"
    assert blocks[0].get("id") == "audio-2"


def test_v0_file_url_with_id() -> None:
    """file/url v0 block with id should convert cleanly."""
    message = HumanMessage(
        content=[
            {
                "type": "file",
                "source_type": "url",
                "url": "https://example.com/doc.pdf",
                "id": "file-1",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "file"
    assert blocks[0].get("id") == "file-1"
    assert blocks[0].get("url") == "https://example.com/doc.pdf"


def test_v0_file_base64_with_id() -> None:
    """file/base64 v0 block with id should convert cleanly."""
    message = HumanMessage(
        content=[
            {
                "type": "file",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "application/pdf",
                "id": "file-2",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "file"
    assert blocks[0].get("id") == "file-2"


def test_v0_image_id_source_still_maps_to_file_id() -> None:
    """source_type='id' (file reference) must still map id to file_id, not block id."""
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "id",
                "id": "file-ref-123",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0].get("file_id") == "file-ref-123"


def test_v0_audio_id_source_still_maps_to_file_id() -> None:
    message = HumanMessage(
        content=[
            {
                "type": "audio",
                "source_type": "id",
                "id": "file-ref-456",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "audio"
    assert blocks[0].get("file_id") == "file-ref-456"


def test_v0_file_id_source_still_maps_to_file_id() -> None:
    message = HumanMessage(
        content=[
            {
                "type": "file",
                "source_type": "id",
                "id": "file-ref-789",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "file"
    assert blocks[0].get("file_id") == "file-ref-789"


def test_v0_block_with_id_preserves_extras() -> None:
    """Genuine extras (non-id unknown keys) must still survive when id is present."""
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/x.png",
        "id": "block-1",
        "alt_text": "An example image",
        "caption": "Example caption",
    }
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result["type"] == "image"
    assert result["id"] == "block-1"
    assert result.get("extras", {}).get("alt_text") == "An example image"
    assert result.get("extras", {}).get("caption") == "Example caption"
