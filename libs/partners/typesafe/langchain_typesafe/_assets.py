"""Send LangChain image content blocks as TypeSafe image assets."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import JsonValue

_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpg"}
_DATA_URL_PREFIX = "data:"
_BASE64_MARKER = ";base64,"


class ImageAssets:
    """Replace image content blocks with TypeSafe asset references.

    TypeSafe replaces an `<image:...>` reference found in any request string with the
    image registered under that reference in the request's `assets`.
    """

    def __init__(self) -> None:
        """Start with no collected assets."""
        self.assets: dict[str, JsonValue] = {}

    def replace(self, value: JsonValue) -> JsonValue:
        """Return `value` with each image content block replaced by its reference.

        Args:
            value: Serialized state or question content.

        Returns:
            The same structure, with image blocks replaced by reference strings.

        Raises:
            ValueError: If an image block has no inline data or uses an unsupported
                format.
        """
        if isinstance(value, dict):
            image = _inline_image(value)
            if image is not None:
                return self._add(*image)
            return {key: self.replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.replace(item) for item in value]
        return value

    def _add(self, mime_type: str, data: str) -> str:
        image_format = _IMAGE_FORMATS.get(mime_type.lower())
        if image_format is None:
            msg = (
                f"TypeSafe does not support image type {mime_type!r}. Supported "
                f"types: {', '.join(sorted(_IMAGE_FORMATS))}."
            )
            raise ValueError(msg)
        reference = f"<image:{uuid4().hex}>"
        self.assets[reference] = {
            "type": "image",
            "format": image_format,
            "data_base64": data,
        }
        return reference


def _inline_image(block: dict[str, Any]) -> tuple[str, str] | None:
    """Return the MIME type and base64 data of an image block, or `None`.

    Recognizes LangChain standard image blocks and OpenAI-format `image_url` blocks,
    which is how images inside serialized messages appear.
    """
    block_type = block.get("type")
    if block_type == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        return _parse_data_url(url)
    if block_type != "image":
        return None
    if isinstance(block.get("base64"), str):
        return _require_mime_type(block), block["base64"]
    if block.get("source_type") == "base64" and isinstance(block.get("data"), str):
        return _require_mime_type(block), block["data"]
    if "url" in block:
        return _parse_data_url(block["url"])
    if "file_id" in block or block.get("source_type") == "id":
        msg = "TypeSafe image blocks must contain base64 data or a `data:` URL."
        raise ValueError(msg)
    return None


def _require_mime_type(block: dict[str, Any]) -> str:
    mime_type = block.get("mime_type")
    if not isinstance(mime_type, str):
        msg = "TypeSafe base64 image blocks require a `mime_type`."
        raise ValueError(msg)  # noqa: TRY004
    return mime_type


def _parse_data_url(url: object) -> tuple[str, str]:
    if (
        not isinstance(url, str)
        or not url.startswith(_DATA_URL_PREFIX)
        or _BASE64_MARKER not in url
    ):
        msg = (
            "TypeSafe image blocks must contain base64 data or a base64 `data:` URL. "
            "Images are not downloaded from other URLs."
        )
        raise ValueError(msg)
    header, data = url[len(_DATA_URL_PREFIX) :].split(_BASE64_MARKER, 1)
    return header, data


__all__ = ["ImageAssets"]
