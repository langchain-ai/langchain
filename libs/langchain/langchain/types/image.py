from typing import Literal, TypedDict


class ImageURL(TypedDict, total=False):
    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image."""

    url: str
    """Either a URL of the image or the base64 encoded image data."""
