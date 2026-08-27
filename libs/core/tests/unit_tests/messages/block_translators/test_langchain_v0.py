from typing import Any

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


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
                "id": "block-1",
            },
            {"type": "image", "url": "https://example.com/image.png", "id": "block-1"},
        ),
        (
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "audio/mpeg",
                "id": "block-2",
            },
            {
                "type": "audio",
                "base64": "<base64 data>",
                "mime_type": "audio/mpeg",
                "id": "block-2",
            },
        ),
        (
            {
                "type": "file",
                "source_type": "text",
                "url": "some text",
                "id": "block-3",
            },
            {
                "type": "text-plain",
                "text": "some text",
                "mime_type": "text/plain",
                "id": "block-3",
            },
        ),
    ],
)
def test_convert_v0_block_carrying_a_block_id(
    block: dict[str, Any], expected: dict[str, Any]
) -> None:
    """A block id must not also be swept into extras and passed twice."""
    assert _convert_legacy_v0_content_block_to_v1(block) == expected


def test_convert_v0_id_source_still_maps_to_file_id() -> None:
    """For `source_type` of `id` the id is the file reference, not a block id."""
    converted = _convert_legacy_v0_content_block_to_v1(
        {"type": "image", "source_type": "id", "id": "file-123"}
    )
    assert converted == {"type": "image", "file_id": "file-123"}
