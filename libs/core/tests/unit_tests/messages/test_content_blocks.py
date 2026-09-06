"""Tests for content block factory helpers."""

import pytest

from langchain_core.messages.content import (
    create_audio_block,
    create_file_block,
    create_image_block,
    create_video_block,
)


@pytest.mark.parametrize(
    "factory",
    [create_image_block, create_video_block, create_audio_block, create_file_block],
)
def test_base64_requires_mime_type(factory: object) -> None:
    """base64 without mime_type must raise for all media factories."""
    with pytest.raises(ValueError, match="mime_type is required when using base64 data"):
        factory(base64="aGk=")  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    [create_image_block, create_video_block, create_audio_block, create_file_block],
)
def test_base64_with_mime_type_accepted(factory: object) -> None:
    block = factory(base64="aGk=", mime_type="image/png")  # type: ignore[operator]
    assert block["base64"] == "aGk="
    assert block["mime_type"] == "image/png"
