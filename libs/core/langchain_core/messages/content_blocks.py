"""Types for content blocks."""

from typing import Literal

from typing_extensions import Required, TypedDict


class DataContentBlock(TypedDict, total=False):
    """Data content block."""

    type: Required[Literal["image", "audio", "file"]]
    """Type of the content block."""
    source_type: Required[Literal["url", "base64", "id", "text"]]
    """Source type."""
    source: Required[str]
    """Data as a URL or data-URI, identifier, or plain-text."""
    mime_type: str
    """MIME type of the content block (if block represents base64 data.)"""
    metadata: dict
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
