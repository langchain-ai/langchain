"""Types for content blocks."""

import warnings
from typing import Any, Literal, Union

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict


# Text and annotations
class UrlCitation(TypedDict, total=False):
    """Citation from a URL."""

    type: Literal["url_citation"]

    url: str
    """Source URL."""

    title: NotRequired[str]
    """Source title."""

    cited_text: NotRequired[str]
    """Text from the source that is being cited."""

    start_index: NotRequired[int]
    """Start index of the response text for which the annotation applies."""

    end_index: NotRequired[int]
    """End index of the response text for which the annotation applies."""


class DocumentCitation(TypedDict, total=False):
    """Annotation for data from a document."""

    type: Literal["document_citation"]

    title: NotRequired[str]
    """Source title."""

    cited_text: NotRequired[str]
    """Text from the source that is being cited."""

    start_index: NotRequired[int]
    """Start index of the response text for which the annotation applies."""

    end_index: NotRequired[int]
    """End index of the response text for which the annotation applies."""


class NonStandardAnnotation(TypedDict, total=False):
    """Provider-specific annotation format."""

    type: Literal["non_standard_annotation"]
    """Type of the content block."""
    value: dict[str, Any]
    """Provider-specific annotation data."""


class TextContentBlock(TypedDict, total=False):
    """Content block for text output."""

    type: Literal["text"]
    """Type of the content block."""
    text: str
    """Block text."""
    annotations: NotRequired[
        list[Union[UrlCitation, DocumentCitation, NonStandardAnnotation]]
    ]
    """Citations and other annotations."""


# Tool calls
class ToolCallContentBlock(TypedDict, total=False):
    """Content block for tool calls.

    These are references to a :class:`~langchain_core.messages.tool.ToolCall` in the
    message's ``tool_calls`` attribute.
    """

    type: Literal["tool_call"]
    """Type of the content block."""
    id: str
    """Tool call ID."""


# Reasoning
class ReasoningContentBlock(TypedDict, total=False):
    """Content block for reasoning output."""

    type: Literal["reasoning"]
    """Type of the content block."""
    reasoning: NotRequired[str]
    """Reasoning text."""


# Multi-modal
class BaseDataContentBlock(TypedDict, total=False):
    """Base class for data content blocks."""

    mime_type: NotRequired[str]
    """MIME type of the content block (if needed)."""


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

_DataContentBlockAdapter: TypeAdapter[DataContentBlock] = TypeAdapter(DataContentBlock)


# Non-standard
class NonStandardContentBlock(TypedDict, total=False):
    """Content block provider-specific data.

    This block contains data for which there is not yet a standard type.
    """

    type: Literal["non_standard"]
    """Type of the content block."""
    value: dict[str, Any]
    """Provider-specific data."""


ContentBlock = Union[
    TextContentBlock,
    ToolCallContentBlock,
    ReasoningContentBlock,
    DataContentBlock,
    NonStandardContentBlock,
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
    try:
        _ = _DataContentBlockAdapter.validate_python(content_block)
    except ValidationError:
        return False
    else:
        return True


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


def convert_to_openai_data_block(block: dict) -> dict:
    """Format standard data content block to format expected by OpenAI."""
    if block["type"] == "image":
        formatted_block = convert_to_openai_image_block(block)

    elif block["type"] == "file":
        if block["source_type"] == "base64":
            file = {"file_data": f"data:{block['mime_type']};base64,{block['data']}"}
            if filename := block.get("filename"):
                file["filename"] = filename
            elif (metadata := block.get("metadata")) and ("filename" in metadata):
                file["filename"] = metadata["filename"]
            else:
                warnings.warn(
                    "OpenAI may require a filename for file inputs. Specify a filename "
                    "in the content block: {'type': 'file', 'source_type': 'base64', "
                    "'mime_type': 'application/pdf', 'data': '...', "
                    "'filename': 'my-pdf'}",
                    stacklevel=1,
                )
            formatted_block = {"type": "file", "file": file}
        elif block["source_type"] == "id":
            formatted_block = {"type": "file", "file": {"file_id": block["id"]}}
        else:
            error_msg = "source_type base64 or id is required for file blocks."
            raise ValueError(error_msg)

    elif block["type"] == "audio":
        if block["source_type"] == "base64":
            audio_format = block["mime_type"].split("/")[-1]
            formatted_block = {
                "type": "input_audio",
                "input_audio": {"data": block["data"], "format": audio_format},
            }
        else:
            error_msg = "source_type base64 is required for audio blocks."
            raise ValueError(error_msg)
    else:
        error_msg = f"Block of type {block['type']} is not supported."
        raise ValueError(error_msg)

    return formatted_block
