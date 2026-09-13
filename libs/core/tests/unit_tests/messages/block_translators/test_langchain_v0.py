import pytest

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


# ---------------------------------------------------------------------------
# Regression tests for https://github.com/langchain-ai/langchain/issues/40424
#
# _convert_legacy_v0_content_block_to_v1 raised
#   TypeError: … got multiple values for keyword argument 'id'
# for any block that carried an `id` field and had source_type in
# {url, base64, text}, because `id` was missing from `known_keys` and
# therefore leaked into `extras`, which was then passed together with the
# explicit `id=block["id"]` keyword argument.
# ---------------------------------------------------------------------------

_ID_BLOCK_CASES = [
    pytest.param(
        {
            "type": "image",
            "source_type": "url",
            "url": "https://example.com/img.png",
            "id": "blk-1",
        },
        {"type": "image", "url": "https://example.com/img.png", "id": "blk-1"},
        id="image/url",
    ),
    pytest.param(
        {
            "type": "image",
            "source_type": "base64",
            "data": "abc123",
            "mime_type": "image/png",
            "id": "blk-2",
        },
        {"type": "image", "base64": "abc123", "mime_type": "image/png", "id": "blk-2"},
        id="image/base64",
    ),
    pytest.param(
        {
            "type": "audio",
            "source_type": "url",
            "url": "https://example.com/audio.mp3",
            "id": "blk-3",
        },
        {"type": "audio", "url": "https://example.com/audio.mp3", "id": "blk-3"},
        id="audio/url",
    ),
    pytest.param(
        {
            "type": "audio",
            "source_type": "base64",
            "data": "abc123",
            "mime_type": "audio/mpeg",
            "id": "blk-4",
        },
        {"type": "audio", "base64": "abc123", "mime_type": "audio/mpeg", "id": "blk-4"},
        id="audio/base64",
    ),
    pytest.param(
        {
            "type": "file",
            "source_type": "url",
            "url": "https://example.com/doc.pdf",
            "id": "blk-5",
        },
        {"type": "file", "url": "https://example.com/doc.pdf", "id": "blk-5"},
        id="file/url",
    ),
    pytest.param(
        {
            "type": "file",
            "source_type": "base64",
            "data": "abc123",
            "mime_type": "application/pdf",
            "id": "blk-6",
        },
        {
            "type": "file",
            "base64": "abc123",
            "mime_type": "application/pdf",
            "id": "blk-6",
        },
        id="file/base64",
    ),
    pytest.param(
        {"type": "file", "source_type": "text", "url": "Hello world", "id": "blk-7"},
        {
            "type": "text-plain",
            "text": "Hello world",
            "mime_type": "text/plain",
            "id": "blk-7",
        },
        id="file/text",
    ),
]


@pytest.mark.parametrize(("block", "expected"), _ID_BLOCK_CASES)
def test_convert_v0_block_with_id_does_not_raise(block: dict, expected: dict) -> None:
    """Regression test for #40424.

    A v0 multimodal block that carries an ``id`` field must be converted
    without raising ``TypeError: got multiple values for keyword argument 'id'``.
    """
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result == expected


@pytest.mark.parametrize(("block", "expected"), _ID_BLOCK_CASES)
def test_content_blocks_property_with_id_does_not_raise(
    block: dict, expected: dict
) -> None:
    """End-to-end regression for #40424: ``message.content_blocks`` must not raise."""
    message = HumanMessage(content=[block])
    result = message.content_blocks
    assert len(result) == 1
    assert result[0] == expected
