import pytest

from langchain_core.messages.content import (
    create_audio_block,
    create_file_block,
    create_image_block,
    create_video_block,
)

BASE64_FACTORIES = [
    create_image_block,
    create_video_block,
    create_audio_block,
    create_file_block,
]


@pytest.mark.parametrize("factory", BASE64_FACTORIES)
def test_base64_without_mime_type_is_rejected(factory: object) -> None:
    with pytest.raises(ValueError, match="mime_type is required"):
        factory(base64="aGk=")  # type: ignore[operator]


@pytest.mark.parametrize("factory", BASE64_FACTORIES)
def test_base64_with_mime_type_is_accepted(factory: object) -> None:
    block = factory(base64="aGk=", mime_type="application/octet-stream")  # type: ignore[operator]
    assert block["base64"] == "aGk="
    assert block["mime_type"] == "application/octet-stream"


@pytest.mark.parametrize("factory", BASE64_FACTORIES)
def test_url_without_mime_type_is_accepted(factory: object) -> None:
    block = factory(url="https://example.com/x")  # type: ignore[operator]
    assert block["url"] == "https://example.com/x"
