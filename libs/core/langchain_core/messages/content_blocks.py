"""Types for content blocks."""

import warnings
from typing import Any, Literal, Union

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict, get_args, get_origin

# Text and annotations


class Citation(TypedDict):
    """Annotation for data from a document."""

    annotation_type: Literal["citation:document"]
    """Type of the content block."""

    url: NotRequired[str]
    """URL of the document source."""

    provenance: NotRequired[str]
    """Provenance of the document, e.g., "Wikipedia", "arXiv", etc."""

    title: NotRequired[str]
    """Source document title."""

    cited_text: NotRequired[str]
    """Text from the document source that is being cited."""

    start_index: NotRequired[int]
    """Start index of the response text for which the annotation applies."""

    end_index: NotRequired[int]
    """End index of the response text for which the annotation applies."""


class NonStandardAnnotation(TypedDict):
    """Provider-specific annotation format."""

    annotation_type: Literal["non_standard"]
    """Type of the content block."""

    value: dict[str, Any]
    """Provider-specific annotation data."""


Annotation = Union[Citation, NonStandardAnnotation]


class TextContentBlock(TypedDict):
    """Content block for text output."""

    block_type: Literal["text"]
    """Type of the content block."""

    text: str
    """Block text."""

    annotations: NotRequired[list[Annotation]]
    """Citations and other annotations."""


# Tool calls
class ToolCallContentBlock(TypedDict):
    """Content block for tool calls.

    These are references to a :class:`~langchain_core.messages.tool.ToolCall` in the
    message's ``tool_calls`` attribute.
    """

    block_type: Literal["tool_call"]
    """Type of the content block."""

    name: str
    """The name of the tool to be called."""

    args: dict[str, Any]
    """The arguments for the tool, as a dictionary."""

    id: str
    """The unique ID for this tool call."""


class ToolOutputContentBlock(TypedDict):
    """Content block for the result of a tool call."""

    block_type: Literal["tool_output"]
    """Type of the content block."""

    output: str  # or `Any`??
    """The output of the tool call."""

    call_id: str
    """The ID of the ToolCallContentBlock it corresponds to."""


# Reasoning
class ReasoningContentBlock(TypedDict):
    """Content block for reasoning output."""

    block_type: Literal["reasoning"]
    """Type of the content block."""

    reasoning_text: NotRequired[str]
    """Reasoning text."""

    effort: NotRequired[str]
    """Reasoning effort level, e.g., 'low', 'medium', 'high'"""

    signature: NotRequired[str]
    """Signature of the reasoning.

    Examples:
    - https://ai.google.dev/gemini-api/docs/thinking#signatures
    """

    annotations: NotRequired[list[Annotation]]
    """Citations and other annotations."""


# Multi-modal
class DataImage(TypedDict):
    """Content block for image data."""

    block_type: Literal["image"]
    """Type of the content block."""

    mime_type: NotRequired[str]
    """MIME type of the image. Required for base64."""

    representation: Literal["url", "base64"]
    """How the data is represented, either as a URL or base64 string."""

    data: str
    """Data as a URL or base64 string."""


class DataAudio(TypedDict):
    """Content block for audio data."""

    block_type: Literal["audio"]
    """Type of the content block."""

    mime_type: NotRequired[str]
    """MIME type of the audio. Required for base64."""

    representation: Literal["url", "base64"]
    """How the data is represented, either as a URL or base64 string."""

    data: str
    """Data as a URL or base64 string."""


class DataText(TypedDict):
    """Content block for plain text data (e.g., from a document)."""

    block_type: Literal["text-data"]
    """Type of the content block."""

    mime_type: Literal["text/plain"]
    """MIME type of the file. Required for base64."""

    representation: Literal["url", "base64"]
    """How the data is represented, either as a URL or base64 string."""

    data: str
    """Data as a URL or base64 string."""


class DataFile(TypedDict):
    """Content block for file data."""

    block_type: Literal["file"]
    """Type of the content block."""

    mime_type: NotRequired[str]
    """MIME type of the file. Required for base64."""

    representation: Literal["url", "base64", "id"]
    """How the data is represented, either as a URL, base64 string, or ID.

    IDs are typically used for files stored in a system like OpenAI's file storage.
    """

    data: str
    """Data as a URL, base64 string, or ID."""


# Non-standard
class NonStandardContentBlock(TypedDict):
    """Content block provider-specific data.

    This block contains data for which there is not yet a standard type.
    """

    block_type: Literal["non_standard"]
    """Type of the content block."""

    value: dict[str, Any]
    """Provider-specific data."""

    tool_calls: NotRequired[list[ToolCallContentBlock]]
    """Tool calls made during reasoning."""

    annotations: NotRequired[list[Annotation]]
    """Citations and other annotations."""


ContentBlock = Union[
    TextContentBlock,
    ToolCallContentBlock,
    ToolOutputContentBlock,
    ReasoningContentBlock,
    NonStandardContentBlock,
    DataImage,
    DataAudio,
    DataText,
    DataFile,
]

DataContentBlock = Union[
    DataImage,
    DataAudio,
    DataText,
    DataFile,
]


def _extract_typedict_block_type_values(union_type: Any) -> set[str]:
    """Extract the values of the 'type' field from a TypedDict union type."""
    result: set[str] = set()
    for value in get_args(union_type):
        annotation = value.__annotations__["block_type"]
        if get_origin(annotation) is Literal:
            result.update(get_args(annotation))
        else:
            msg = f"{value} 'block_type' is not a Literal"
            raise ValueError(msg)
    return result


KNOWN_BLOCK_TYPES = {
    bt
    for bt in get_args(ContentBlock)
    for bt in get_args(bt.__annotations__["block_type"])
}

# Adapter for DataContentBlock
_DataAdapter: TypeAdapter[DataContentBlock] = TypeAdapter(DataContentBlock)


def is_data_block(block: dict) -> bool:
    """Check if the content block is a standard data content block.

    Args:
        block: The content block to check.

    Returns:
        True if the content block is a data content block, False otherwise.
    """
    try:
        _DataAdapter.validate_python(block)
    except ValidationError:
        return False
    else:
        return True


# These would need to be refactored
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
