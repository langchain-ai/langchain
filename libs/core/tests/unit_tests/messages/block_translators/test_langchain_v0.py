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
    ("block_type", "source"),
    [
        ("image", {"source_type": "url", "url": "https://example.com/x.png"}),
        (
            "image",
            {"source_type": "base64", "data": "aGk=", "mime_type": "image/png"},
        ),
        ("audio", {"source_type": "url", "url": "https://example.com/x.mp3"}),
        (
            "audio",
            {"source_type": "base64", "data": "aGk=", "mime_type": "audio/mp3"},
        ),
        ("file", {"source_type": "url", "url": "https://example.com/x.pdf"}),
        (
            "file",
            {"source_type": "base64", "data": "aGk=", "mime_type": "application/pdf"},
        ),
    ],
    ids=[
        "image-url",
        "image-base64",
        "audio-url",
        "audio-base64",
        "file-url",
        "file-base64",
    ],
)
def test_block_id_is_not_treated_as_an_extra(
    block_type: str, source: dict[str, str]
) -> None:
    """Test a v0 block carrying `id` converts instead of raising.

    Regression test: `_extract_v0_extras` filtered by each branch's `known_keys`, and
    the `url`/`base64` branches did not list `id`. It therefore stayed in `extras`,
    and expanding `**extras` alongside the explicit `id=block["id"]` argument raised
    `TypeError: got multiple values for keyword argument 'id'`. The `source_type="id"`
    branches were unaffected because their `known_keys` included it.
    """
    block = {"type": block_type, "id": "block-1", **source}

    converted = _convert_legacy_v0_content_block_to_v1(block)

    assert converted["id"] == "block-1"
    assert "id" not in converted.get("extras", {})


def test_block_id_does_not_shadow_real_extras() -> None:
    """Test excluding `id` from extras leaves genuine extra keys intact."""
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/x.png",
        "id": "block-1",
        "alt_text": "kept",
    }

    converted = _convert_legacy_v0_content_block_to_v1(block)

    assert converted["id"] == "block-1"
    assert converted["extras"] == {"alt_text": "kept"}
