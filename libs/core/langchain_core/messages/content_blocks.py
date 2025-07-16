"""Types for content blocks."""

import warnings
from typing import Any, Literal, Optional, Union

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict, get_args, get_origin


# Text and annotations
class UrlCitation(TypedDict):
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


class DocumentCitation(TypedDict):
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


class NonStandardAnnotation(TypedDict):
    """Provider-specific annotation format."""

    type: Literal["non_standard_annotation"]
    """Type of the content block."""
    value: dict[str, Any]
    """Provider-specific annotation data."""


class TextContentBlock(TypedDict):
    """Content block for text output."""

    type: Literal["text"]
    """Type of the content block."""
    text: str
    """Block text."""
    annotations: NotRequired[
        list[Union[UrlCitation, DocumentCitation, NonStandardAnnotation]]
    ]
    """Citations and other annotations."""


def make_text_block(
    text: str, annotations: Optional[list[dict[str, Any]]] = None
) -> dict[str, Any]:
    """Return a dict matching TextContentBlock.

    {
        "type": "text",
        "text": <text>,
        "annotations": [ ... ]  # optional
    }
    """
    block: dict[str, Any] = {
        "type": "text",
        "text": text,
    }
    if annotations is not None:
        block["annotations"] = annotations
    return block


# Tool calls
class ToolCallContentBlock(TypedDict):
    """Content block for tool calls.

    These are references to a :class:`~langchain_core.messages.tool.ToolCall` in the
    message's ``tool_calls`` attribute.
    """

    type: Literal["tool_call"]
    """Type of the content block."""
    id: str
    """Tool call ID."""


def make_tool_call_block(
    tool_call_id: str,
) -> dict[str, Any]:
    """Return a dict matching ToolCallContentBlock.

    {
        "type": "tool_call",
        "id": <tool_call_id>
    }
    """
    return {
        "type": "tool_call",
        "id": tool_call_id,
    }


# Reasoning
class ReasoningContentBlock(TypedDict):
    """Content block for reasoning output."""

    type: Literal["reasoning"]
    """Type of the content block."""
    reasoning: NotRequired[str]
    """Reasoning text."""


def make_reasoning_block(
    reasoning: Optional[str] = None,
) -> dict[str, Any]:
    """Return a dict matching ReasoningContentBlock.

    {
        "type": "reasoning",
        "reasoning": <reasoning>  # optional
    }
    """
    block: dict[str, Any] = {"type": "reasoning"}
    if reasoning is not None:
        block["reasoning"] = reasoning
    return block


# Multi-modal
class BaseDataContentBlock(TypedDict):
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


def make_url_content_block(
    url: str,
    mime_type: Optional[str] = None,
    type_: Literal["image", "audio", "file"] = "file",
) -> dict[str, Any]:
    """Return a dict matching URLContentBlock.

    {
        "type": <type>,
        "source_type": "url",
        "url": <url>,
        "mime_type": <mime_type>  # optional
    }
    """
    block: dict[str, Any] = {
        "type": type_,
        "source_type": "url",
        "url": url,
    }
    if mime_type is not None:
        block["mime_type"] = mime_type
    return block


class Base64ContentBlock(BaseDataContentBlock):
    """Content block for inline data from a base64 string."""

    type: Literal["image", "audio", "file"]
    """Type of the content block."""
    source_type: Literal["base64"]
    """Source type (base64)."""
    data: str
    """Data as a base64 string."""


def make_base64_content_block(
    data: str,
    mime_type: Optional[str] = None,
    type_: Literal["image", "audio", "file"] = "file",
) -> dict[str, Any]:
    """Return a dict matching Base64ContentBlock.

    {
        "type": <type>,
        "source_type": "base64",
        "data": <base64_data>,
        "mime_type": <mime_type>  # optional
    }
    """
    block: dict[str, Any] = {
        "type": type_,
        "source_type": "base64",
        "data": data,
    }
    if mime_type is not None:
        block["mime_type"] = mime_type
    return block


class PlainTextContentBlock(BaseDataContentBlock):
    """Content block for plain text data (e.g., from a document)."""

    type: Literal["file"]
    """Type of the content block."""
    source_type: Literal["text"]
    """Source type (text)."""
    text: str
    """Text data."""


def make_plain_text_content_block(
    text: str,
    mime_type: Optional[str] = None,
) -> dict[str, Any]:
    """Return a dict matching PlainTextContentBlock.

    {
        "type": "file",
        "source_type": "text",
        "text": <text>,
        "mime_type": <mime_type>  # optional
    }
    """
    block: dict[str, Any] = {
        "type": "file",
        "source_type": "text",
        "text": text,
    }
    if mime_type is not None:
        block["mime_type"] = mime_type
    return block


class IDContentBlock(BaseDataContentBlock):
    """Content block for data specified by an identifier."""

    type: Literal["image", "audio", "file"]
    """Type of the content block."""
    source_type: Literal["id"]
    """Source type (id)."""
    id: str
    """Identifier for data source."""


def make_id_content_block(
    id_: str,
    mime_type: Optional[str] = None,
    type_: Literal["image", "audio", "file"] = "file",
) -> dict[str, Any]:
    """Return a dict matching IDContentBlock.

    {
        "type": <type>,
        "source_type": "id",
        "id": <id>,
        "mime_type": <mime_type>  # optional
    }
    """
    block: dict[str, Any] = {
        "type": type_,
        "source_type": "id",
        "id": id_,
    }
    if mime_type is not None:
        block["mime_type"] = mime_type
    return block


DataContentBlock = Union[
    URLContentBlock,
    Base64ContentBlock,
    PlainTextContentBlock,
    IDContentBlock,
]

_DataContentBlockAdapter: TypeAdapter[DataContentBlock] = TypeAdapter(DataContentBlock)


# Non-standard
class NonStandardContentBlock(TypedDict):
    """Content block provider-specific data.

    This block contains data for which there is not yet a standard type.
    """

    type: Literal["non_standard"]
    """Type of the content block."""
    value: dict[str, Any]
    """Provider-specific data."""


def make_non_standard_content_block(value: dict[str, Any]) -> dict[str, Any]:
    """Return a dict matching NonStandardContentBlock.

    {
        "type": "non_standard",
        "value": <value>
    }
    """
    return {
        "type": "non_standard",
        "value": value,
    }


ContentBlock = Union[
    TextContentBlock,
    ToolCallContentBlock,
    ReasoningContentBlock,
    DataContentBlock,
    NonStandardContentBlock,
]


def _extract_typedict_type_values(union_type: Any) -> set[str]:
    """Extract the values of the 'type' field from a TypedDict union type."""
    result: set[str] = set()
    for value in get_args(union_type):
        annotation = value.__annotations__["type"]
        if get_origin(annotation) is Literal:
            result.update(get_args(annotation))
        else:
            msg = f"{value} 'type' is not a Literal"
            raise ValueError(msg)
    return result


# {"text", "tool_call", "reasoning", "non_standard", "image", "audio", "file"}
KNOWN_BLOCK_TYPES = _extract_typedict_type_values(ContentBlock)


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
