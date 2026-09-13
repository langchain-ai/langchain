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
    ("block", "expected_id"),
    [
        # image-url with block id
        (
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
                "id": "block-img-url",
            },
            "block-img-url",
        ),
        # image-base64 with block id
        (
            {
                "type": "image",
                "source_type": "base64",
                "data": "<b64>",
                "mime_type": "image/png",
                "id": "block-img-b64",
            },
            "block-img-b64",
        ),
        # audio-url with block id
        (
            {
                "type": "audio",
                "source_type": "url",
                "url": "https://example.com/audio.mp3",
                "id": "block-aud-url",
            },
            "block-aud-url",
        ),
        # audio-base64 with block id
        (
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<b64>",
                "mime_type": "audio/mpeg",
                "id": "block-aud-b64",
            },
            "block-aud-b64",
        ),
        # file-url with block id
        (
            {
                "type": "file",
                "source_type": "url",
                "url": "https://example.com/doc.pdf",
                "id": "block-file-url",
            },
            "block-file-url",
        ),
        # file-base64 with block id
        (
            {
                "type": "file",
                "source_type": "base64",
                "data": "<b64>",
                "mime_type": "application/pdf",
                "id": "block-file-b64",
            },
            "block-file-b64",
        ),
        # file-text with block id
        (
            {
                "type": "file",
                "source_type": "text",
                "url": "hello world",
                "id": "block-file-text",
            },
            "block-file-text",
        ),
    ],
)
def test_convert_v0_block_with_id_no_duplicate_kwarg(
    block: dict, expected_id: str
) -> None:
    """Regression test for #40424.

    When a v0 content block carries an ``"id"`` field, the translator must not
    raise ``TypeError: got multiple values for keyword argument 'id'``.  The
    ``"id"`` key must be in ``known_keys`` so it is excluded from ``extras``
    and only forwarded once via the explicit ``id=`` parameter.
    """
    # Should not raise TypeError
    result = _convert_legacy_v0_content_block_to_v1(block)
    assert result.get("id") == expected_id
