"""Types for content blocks."""

from typing import Any, Literal, Union

from typing_extensions import NotRequired, TypedDict


class BaseDataContentBlock(TypedDict):
    """Base class for data content blocks."""

    mime_type: NotRequired[str]
    """MIME type of the content block (if needed)."""
    metadata: NotRequired[dict]
    """Provider-specific metadata such as citations or filenames."""


class URLContentBlock(BaseDataContentBlock):
    """Content block for data from a URL."""

    type: Literal["image", "audio", "file"]
    """Type of the content block."""
    source_type: Literal["url"]
    """Source type (url)."""
    url: str
    """URL for data."""


class Base64ContentBlock(BaseDataContentBlock):
    """Content block for inline data from a base64 string."""

    type: Literal["image", "audio", "file"]
    """Type of the content block."""
    source_type: Literal["base64"]
    """Source type (base64)."""
    data: str
    """Data as a base64 string."""


class PlainTextContentBlock(BaseDataContentBlock):
    """Content block for plain text data (e.g., from a document)."""

    type: Literal["file"]
    """Type of the content block."""
    source_type: Literal["text"]
    """Source type (text)."""
    text: str
    """Text data."""


class IDContentBlock(TypedDict):
    """Content block for data specified by an identifier."""

    type: Literal["image", "audio", "file"]
    """Type of the content block."""
    source_type: Literal["id"]
    """Source type (id)."""
    id: str
    """Identifier for data source."""


DataContentBlock = Union[
    URLContentBlock,
    Base64ContentBlock,
    PlainTextContentBlock,
    IDContentBlock,
]


def is_data_content_block(
    content_block: dict,
) -> bool:
    """Check if the content block is a standard data content block.

    Args:
        content_block: The content block to check.

    Returns:
        True if the content block is a data content block, False otherwise.
    """
    required_keys = {"type", "source_type"}
    if all(required_key in content_block for required_key in required_keys):
        if content_block["source_type"] == "url":
            return "url" in content_block
        if content_block["source_type"] == "base64":
            return "data" in content_block
        if content_block["source_type"] == "text":
            return "text" in content_block
        if content_block["source_type"] == "id":
            return "id" in content_block
        return False
    return False


def convert_to_openai_image_block(content_block: dict[str, Any]) -> dict:
    """Convert image content block to format expected by OpenAI Chat Completions API."""
    if content_block["source_type"] == "url":
        return {
            "type": "image_url",
            "image_url": {
                "url": content_block["url"],
            },
        }
    if content_block["source_type"] == "base64":
        if "mime_type" not in content_block:
            error_message = "mime_type key is required for base64 data."
            raise ValueError(error_message)
        mime_type = content_block["mime_type"]
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{content_block['data']}",
            },
        }
    error_message = "Unsupported source type. Only 'url' and 'base64' are supported."
    raise ValueError(error_message)
