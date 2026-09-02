from collections.abc import Callable

import pytest

from langchain_core.messages import content as types

BlockFactory = Callable[..., dict]

CREATE_CASES = [
    pytest.param("image", types.create_image_block, id="image"),
    pytest.param("video", types.create_video_block, id="video"),
    pytest.param("audio", types.create_audio_block, id="audio"),
    pytest.param("file", types.create_file_block, id="file"),
]

MIME_TYPES = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/mpeg",
    "file": "application/pdf",
}


@pytest.mark.parametrize(("_type", "factory"), CREATE_CASES)
def test_base64_without_mime_type_raises(_type: str, factory: BlockFactory) -> None:
    msg = "mime_type is required when using base64 data"
    with pytest.raises(ValueError, match=msg):
        factory(base64="aGVsbG8=")


@pytest.mark.parametrize(("_type", "factory"), CREATE_CASES)
def test_base64_with_mime_type(_type: str, factory: BlockFactory) -> None:
    block = factory(base64="aGVsbG8=", mime_type=MIME_TYPES[_type])
    assert block["base64"] == "aGVsbG8="
    assert block["mime_type"] == MIME_TYPES[_type]


@pytest.mark.parametrize(("_type", "factory"), CREATE_CASES)
def test_no_source_raises(_type: str, factory: BlockFactory) -> None:
    msg = "Must provide one of: url, base64, or file_id"
    with pytest.raises(ValueError, match=msg):
        factory()
