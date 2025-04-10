"""Types for content blocks."""

from typing import Literal

from typing_extensions import NotRequired, TypedDict


class DataContentBlock(TypedDict):
    """Data content block."""

    type: Literal["image", "audio", "file"]
    """Type of the content block."""
    source_type: Literal["url", "base64", "id", "text"]
    """Source type."""
    source: str
    """Data as a URL or data-URI, identifier, or plain-text."""
    mime_type: NotRequired[str]
    """MIME type of the content block (if block represents base64 data.)"""
    metadata: NotRequired[dict]
    """Provider-specific metadata such as citations or filenames."""


def is_data_content_block(
    content_block: dict,
) -> bool:
    """Check if the content block is a data content block.

    Args:
        content_block: The content block to check.

    Returns:
        True if the content block is a data content block, False otherwise.
    """
    required_keys = DataContentBlock.__required_keys__
    return all(required_key in content_block for required_key in required_keys)


def convert_image_content_block_to_image_url(content_block: DataContentBlock) -> dict:
    """Convert image content block to format expected by OpenAI Chat Completions API."""
    if content_block["source_type"] == "url":
        return {
            "type": "image_url",
            "image_url": {
                "url": content_block["source"],
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
                "url": f"data:{mime_type};base64,{content_block['source']}",
            },
        }
    error_message = "Unsupported source type. Only 'url' and 'base64' are supported."
    raise ValueError(error_message)
