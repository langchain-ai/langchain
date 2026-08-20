"""Tests for the content block factory functions."""

from typing import Any

import pytest

from langchain_core.messages.content import (
    create_audio_block,
    create_file_block,
    create_image_block,
    create_video_block,
)

_SOURCE_FACTORIES = [
    create_image_block,
    create_video_block,
    create_audio_block,
    create_file_block,
]
_FACTORY_IDS = ["image", "video", "audio", "file"]


@pytest.mark.parametrize("factory", _SOURCE_FACTORIES, ids=_FACTORY_IDS)
def test_base64_requires_mime_type(factory: Any) -> None:
    """Test every source-bearing factory rejects `base64` without `mime_type`.

    Each factory documents `ValueError: ... if base64 is used without mime_type`, but
    `create_image_block` did not enforce it, so an image block could be built with
    base64 data and no MIME type while the other three rejected the same call.
    """
    with pytest.raises(ValueError, match="mime_type is required"):
        factory(base64="aGk=")


@pytest.mark.parametrize("factory", _SOURCE_FACTORIES, ids=_FACTORY_IDS)
def test_base64_with_mime_type_is_accepted(factory: Any) -> None:
    """Test supplying `mime_type` alongside `base64` builds a block."""
    block = factory(base64="aGk=", mime_type="application/octet-stream")

    assert block["base64"] == "aGk="
    assert block["mime_type"] == "application/octet-stream"


@pytest.mark.parametrize("factory", _SOURCE_FACTORIES, ids=_FACTORY_IDS)
def test_requires_a_source(factory: Any) -> None:
    """Test every factory requires one of `url`, `base64` or `file_id`."""
    with pytest.raises(ValueError, match="Must provide one of"):
        factory()


@pytest.mark.parametrize("factory", _SOURCE_FACTORIES, ids=_FACTORY_IDS)
@pytest.mark.parametrize(
    "source",
    [{"url": "https://example.com/x"}, {"file_id": "file-1"}],
    ids=["url", "file_id"],
)
def test_non_base64_sources_need_no_mime_type(
    factory: Any, source: dict[str, str]
) -> None:
    """Test `url` and `file_id` sources are accepted without a `mime_type`."""
    block = factory(**source)

    key, value = next(iter(source.items()))
    assert block[key] == value
    assert "mime_type" not in block
