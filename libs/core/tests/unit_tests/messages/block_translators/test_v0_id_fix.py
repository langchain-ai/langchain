"""Regression tests for v0 multimodal blocks carrying an ``id`` field.

Covers the fix for https://github.com/langchain-ai/langchain/issues/40424:
``_convert_legacy_v0_content_block_to_v1`` raised ``TypeError`` when a v0
block included ``id`` because the key was missing from ``known_keys`` and
ended up in ``**extras``, colliding with the explicit ``id=block["id"]``
kwarg.

Seven branches are affected: image/url, image/base64, audio/url,
audio/base64, file/url, file/base64, and file/text.
"""

import pytest

from langchain_core.messages import HumanMessage
from langchain_core.messages.block_translators.langchain_v0 import (
    _convert_legacy_v0_content_block_to_v1,
)


# ---------- parametrized: each branch with ``id`` present ---------- #


@pytest.mark.parametrize(
    "description, v0_block, expected_id",
    [
        (
            "image/url",
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
                "id": "img-url-1",
            },
            "img-url-1",
        ),
        (
            "image/base64",
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "image/png",
                "id": "img-b64-1",
            },
            "img-b64-1",
        ),
        (
            "audio/url",
            {
                "type": "audio",
                "source_type": "url",
                "url": "https://example.com/audio.mp3",
                "id": "aud-url-1",
            },
            "aud-url-1",
        ),
        (
            "audio/base64",
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "audio/mpeg",
                "id": "aud-b64-1",
            },
            "aud-b64-1",
        ),
        (
            "file/url",
            {
                "type": "file",
                "source_type": "url",
                "url": "https://example.com/doc.pdf",
                "id": "file-url-1",
            },
            "file-url-1",
        ),
        (
            "file/base64",
            {
                "type": "file",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "application/pdf",
                "id": "file-b64-1",
            },
            "file-b64-1",
        ),
        (
            "file/text",
            {
                "type": "file",
                "source_type": "text",
                "url": "inline text content",
                "id": "file-txt-1",
            },
            "file-txt-1",
        ),
    ],
    ids=lambda val: val if isinstance(val, str) and "/" in val else "",
)
def test_v0_block_with_id_converts_without_error(
    description: str, v0_block: dict, expected_id: str
) -> None:
    """Blocks carrying ``id`` must not raise TypeError."""
    result = _convert_legacy_v0_content_block_to_v1(v0_block)
    assert result["id"] == expected_id, (
        f"{description}: id mismatch, got {result.get('id')}"
    )


# ---------- parametrized: each branch WITHOUT ``id`` (regression) ---------- #


@pytest.mark.parametrize(
    "description, v0_block",
    [
        (
            "image/url",
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
            },
        ),
        (
            "image/base64",
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "image/png",
            },
        ),
        (
            "audio/url",
            {
                "type": "audio",
                "source_type": "url",
                "url": "https://example.com/audio.mp3",
            },
        ),
        (
            "audio/base64",
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "audio/mpeg",
            },
        ),
        (
            "file/url",
            {
                "type": "file",
                "source_type": "url",
                "url": "https://example.com/doc.pdf",
            },
        ),
        (
            "file/base64",
            {
                "type": "file",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "application/pdf",
            },
        ),
        (
            "file/text",
            {
                "type": "file",
                "source_type": "text",
                "url": "inline text content",
            },
        ),
    ],
    ids=lambda val: val if isinstance(val, str) and "/" in val else "",
)
def test_v0_block_without_id_still_converts(
    description: str, v0_block: dict
) -> None:
    """Blocks *without* ``id`` must keep working after the fix."""
    result = _convert_legacy_v0_content_block_to_v1(v0_block)
    # These blocks were constructed without an ID, so the conversion should
    # not inject one (the id-less path doesn't call the create_* helpers).
    assert "id" not in result or result.get("id") is None, (
        f"{description}: unexpected id in result"
    )


# ---------- end-to-end through HumanMessage.content_blocks ---------- #


def test_content_blocks_with_id_end_to_end() -> None:
    """The original reproducer from the issue must work end-to-end."""
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
                "id": "block-1",
            }
        ]
    )
    blocks = message.content_blocks
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["id"] == "block-1"
