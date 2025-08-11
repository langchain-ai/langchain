"""Standard, multimodal content blocks for Large Language Model I/O.

.. warning::
    This module is under active development. The API is unstable and subject to
    change in future releases.

This module provides a standardized data structure for representing inputs to and
outputs from LLMs. The core abstraction is the **Content Block**, a ``TypedDict`` that
can represent a piece of text, an image, a tool call, or other structured data.

**Rationale**

Different LLM providers use distinct and incompatible API schemas. This module
introduces a unified, provider-agnostic format to standardize these interactions. A
message to or from a model is simply a ``list`` of ``ContentBlock`` objects, allowing
for the natural interleaving of text, images, and other content in a single, ordered
sequence.

An adapter for a specific provider is responsible for translating this standard list of
blocks into the format required by its API.

**Extensibility**

Data **not yet mapped** to a standard block may be represented using the
``NonStandardContentBlock``, which allows for provider-specific data to be included
without losing the benefits of type checking and validation.

Furthermore, provider-specific fields **within** a standard block are fully supported
by default in the ``extras`` field of each block. This allows for additional metadata
to be included without breaking the standard structure.

Following widespread adoption of `PEP 728 <https://peps.python.org/pep-0728/>`__, we will add
``extra_items=Any`` as a param to Content Blocks. This will signify to type checkers
that additional provider-specific fields are allowed outside of the ``extras`` field,
and that will become the new standard approach to adding provider-specific metadata.

.. warning::
    Do not heavily rely on the ``extras`` field for provider-specific data! This field
    is subject to deprecation in future releases as we move towards PEP 728.

**Example with PEP 728 provider-specific fields:**

.. code-block:: python

    # Content block definition
    # NOTE: `extra_items=Any`
    class TextContentBlock(TypedDict, extra_items=Any):
        type: Literal["text"]
        id: NotRequired[str]
        text: str
        annotations: NotRequired[list[Annotation]]
        index: NotRequired[int]

.. code-block:: python

    from langchain_core.messages.content_blocks import TextContentBlock

    # Create a text content block with provider-specific fields
    my_block: TextContentBlock = {
        # Add required fields
        "type": "text",
        "text": "Hello, world!",
        # Additional fields not specified in the TypedDict
        # These are valid with PEP 728 and are typed as Any
        "openai_metadata": {"model": "gpt-4", "temperature": 0.7},
        "anthropic_usage": {"input_tokens": 10, "output_tokens": 20},
        "custom_field": "any value",
    }

    # Mutating an existing block to add provider-specific fields
    openai_data = my_block["openai_metadata"]  # Type: Any

.. note::
    PEP 728 is enabled with ``# type: ignore[call-arg]`` comments to suppress warnings
    from type checkers that don't yet support it. The functionality works correctly
    in Python 3.13+ and will be fully supported as the ecosystem catches up.

**Key Block Types**

The module defines several types of content blocks, including:

- ``TextContentBlock``: Standard text.
- ``ImageContentBlock``, ``Audio...``, ``Video...``, ``PlainText...``, ``File...``: For multimodal data.
- ``ToolCallContentBlock``: For function calling.
- ``ReasoningContentBlock``: To capture a model's thought process.
- ``Citation``: For annotations that link generated text to a source document.

**Example Usage**

.. code-block:: python

    # Direct construction:
    from langchain_core.messages.content_blocks import TextContentBlock, ImageContentBlock

    multimodal_message: AIMessage = [
        TextContentBlock(type="text", text="What is shown in this image?"),
        ImageContentBlock(
            type="image",
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]

    from langchain_core.messages.content_blocks import create_text_block, create_image_block

    # Using factory functions:
    multimodal_message: AIMessage = [
        create_text_block("What is shown in this image?"),
        create_image_block(
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]

Factory functions like ``create_text_block`` and ``create_image_block`` are provided
and offer benefits such as:
- Automatic ID generation (when not provided)
- No need to manually specify the ``type`` field

"""  # noqa: E501

import warnings
from typing import Any, Literal, Optional, Union
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict, TypeGuard


def _ensure_id(id_val: Optional[str]) -> str:
    """Ensure the ID is a valid string, generating a new UUID if not provided.

    Auto-generated UUIDs are prefixed by ``'lc_'`` to indicate they are
    LangChain-generated IDs.

    Args:
        id_val: Optional string ID value to validate.

    Returns:
        A valid string ID, either the provided value or a new UUID.
    """
    return id_val or str(f"lc_{uuid4()}")


class Citation(TypedDict):
    """Annotation for citing data from a document.

    .. note::
        ``start``/``end`` indices refer to the **response text**,
        not the source text. This means that the indices are relative to the model's
        response, not the original document (as specified in the ``url``).

    .. note::
        ``create_citation`` may also be used as a factory to create a ``Citation``.
        Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["citation"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    url: NotRequired[str]
    """URL of the document source."""

    # For future consideration, if needed:
    # provenance: NotRequired[str]
    # """Provenance of the document, e.g., ``'Wikipedia'``, ``'arXiv'``, etc.

    # Included for future compatibility; not currently implemented.
    # """

    title: NotRequired[str]
    """Source document title.

    For example, the page title for a web page or the title of a paper.
    """

    start_index: NotRequired[int]
    """Start index of the **response text** (``TextContentBlock.text``) for which the
    annotation applies."""

    end_index: NotRequired[int]
    """End index of the **response text** (``TextContentBlock.text``) for which the
    annotation applies."""

    cited_text: NotRequired[str]
    """Excerpt of source text being cited."""

    # NOTE: not including spans for the raw document text (such as `text_start_index`
    # and `text_end_index`) as this is not currently supported by any provider. The
    # thinking is that the `cited_text` should be sufficient for most use cases, and it
    # is difficult to reliably extract spans from the raw document text across file
    # formats or encoding schemes.

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class NonStandardAnnotation(TypedDict):
    """Provider-specific annotation format."""

    type: Literal["non_standard_annotation"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    value: dict[str, Any]
    """Provider-specific annotation data."""


Annotation = Union[Citation, NonStandardAnnotation]


class TextContentBlock(TypedDict):
    """Text output from a LLM.

    This typically represents the main text content of a message, such as the response
    from a language model or the text of a user message.

    .. note::
        ``create_text_block`` may also be used as a factory to create a
        ``TextContentBlock``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["text"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    text: str
    """Block text."""

    annotations: NotRequired[list[Annotation]]
    """``Citation``s and other annotations."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ToolCall(TypedDict):
    """Represents a request to call a tool.

    Example:

        .. code-block:: python

            {
                "name": "foo",
                "args": {"a": 1},
                "id": "123"
            }

        This represents a request to call the tool named "foo" with arguments {"a": 1}
        and an identifier of "123".

    .. note::
        ``create_tool_call`` may also be used as a factory to create a
        ``ToolCall``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["tool_call"]
    """Used for discrimination."""

    id: Optional[str]
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """
    # TODO: Consider making this NotRequired[str] in the future.

    name: str
    """The name of the tool to be called."""

    args: dict[str, Any]
    """The arguments to the tool call."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ToolCallChunk(TypedDict):
    """A chunk of a tool call (e.g., as part of a stream).

    When merging ``ToolCallChunks`` (e.g., via ``AIMessageChunk.__add__``),
    all string attributes are concatenated. Chunks are only merged if their
    values of ``index`` are equal and not ``None``.

    Example:

    .. code-block:: python

        left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
        right_chunks = [ToolCallChunk(name=None, args='1}', index=0)]

        (
            AIMessageChunk(content="", tool_call_chunks=left_chunks)
            + AIMessageChunk(content="", tool_call_chunks=right_chunks)
        ).tool_call_chunks == [ToolCallChunk(name='foo', args='{"a":1}', index=0)]
    """

    # TODO: Consider making fields NotRequired[str] in the future.

    type: Literal["tool_call_chunk"]
    """Used for serialization."""

    id: Optional[str]
    """An identifier associated with the tool call."""

    name: Optional[str]
    """The name of the tool to be called."""

    args: Optional[str]
    """The arguments to the tool call."""

    index: Optional[int]
    """The index of the tool call in a sequence."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class InvalidToolCall(TypedDict):
    """Allowance for errors made by LLM.

    Here we add an ``error`` key to surface errors made during generation
    (e.g., invalid JSON arguments.)
    """

    # TODO: Consider making fields NotRequired[str] in the future.

    type: Literal["invalid_tool_call"]
    """Used for discrimination."""

    id: Optional[str]
    """An identifier associated with the tool call."""

    name: Optional[str]
    """The name of the tool to be called."""

    args: Optional[str]
    """The arguments to the tool call."""

    error: Optional[str]
    """An error message associated with the tool call."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


# Note: These are not standard tool calls, but rather provider-specific built-in tools.
# Web search
class WebSearchCall(TypedDict):
    """Built-in web search tool call."""

    type: Literal["web_search_call"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    query: NotRequired[str]
    """The search query used in the web search tool call."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class WebSearchResult(TypedDict):
    """Result of a built-in web search tool call."""

    type: Literal["web_search_result"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    urls: NotRequired[list[str]]
    """List of URLs returned by the web search tool call."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class CodeInterpreterCall(TypedDict):
    """Built-in code interpreter tool call."""

    type: Literal["code_interpreter_call"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    language: NotRequired[str]
    """The name of the programming language used in the code interpreter tool call."""

    code: NotRequired[str]
    """The code to be executed by the code interpreter."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class CodeInterpreterOutput(TypedDict):
    """Output of a singular code interpreter tool call.

    Full output of a code interpreter tool call is represented by
    ``CodeInterpreterResult`` which is a list of these blocks.
    """

    type: Literal["code_interpreter_output"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    return_code: NotRequired[int]
    """Return code of the executed code.

    Example: ``0`` for success, non-zero for failure.
    """

    stderr: NotRequired[str]
    """Standard error output of the executed code."""

    stdout: NotRequired[str]
    """Standard output of the executed code."""

    file_ids: NotRequired[list[str]]
    """List of file IDs generated by the code interpreter."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class CodeInterpreterResult(TypedDict):
    """Result of a code interpreter tool call."""

    type: Literal["code_interpreter_result"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    output: list[CodeInterpreterOutput]
    """List of outputs from the code interpreter tool call."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ReasoningContentBlock(TypedDict):
    """Reasoning output from a LLM.

    .. note::
        ``create_reasoning_block`` may also be used as a factory to create a
        ``ReasoningContentBlock``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["reasoning"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    reasoning: NotRequired[str]
    """Reasoning text.

    Either the thought summary or the raw reasoning text itself. This is often parsed
    from ``<think>`` tags in the model's response.
    """

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


# Note: `title` and `context` are fields that could be used to provide additional
# information about the file, such as a description or summary of its content.
# E.g. with Claude, you can provide a context for a file which is passed to the model.
class ImageContentBlock(TypedDict):
    """Image data.

    .. note::
        ``create_image_block`` may also be used as a factory to create a
        ``ImageContentBlock``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["image"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    file_id: NotRequired[str]
    """ID of the image file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the image. Required for base64.

    `Examples from IANA <https://www.iana.org/assignments/media-types/media-types.xhtml#image>`__
    """

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the image."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class VideoContentBlock(TypedDict):
    """Video data.

    .. note::
        ``create_video_block`` may also be used as a factory to create a
        ``VideoContentBlock``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["video"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    file_id: NotRequired[str]
    """ID of the video file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the video. Required for base64.

    `Examples from IANA <https://www.iana.org/assignments/media-types/media-types.xhtml#video>`__
    """

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the video."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class AudioContentBlock(TypedDict):
    """Audio data.

    .. note::
        ``create_audio_block`` may also be used as a factory to create an
        ``AudioContentBlock``. Benefits include:
        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["audio"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    file_id: NotRequired[str]
    """ID of the audio file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the audio. Required for base64.

    `Examples from IANA <https://www.iana.org/assignments/media-types/media-types.xhtml#audio>`__

    """

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the audio."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class PlainTextContentBlock(TypedDict):
    """Plaintext data (e.g., from a document).

    .. note::
        Title and context are optional fields that may be passed to the model. See
        Anthropic `example <https://docs.anthropic.com/en/docs/build-with-claude/citations#citable-vs-non-citable-content>`__.

    .. note::
        ``create_plaintext_block`` may also be used as a factory to create a
        ``PlainTextContentBlock``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["text-plain"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    file_id: NotRequired[str]
    """ID of the plaintext file, e.g., from a file storage system."""

    mime_type: Literal["text/plain"]
    """MIME type of the file. Required for base64."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the plaintext."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    text: NotRequired[str]
    """Plaintext content. This is optional if the data is provided as base64."""

    title: NotRequired[str]
    """Title of the text data, e.g., the title of a document."""

    context: NotRequired[str]
    """Context for the text, e.g., a description or summary of the text's content."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class FileContentBlock(TypedDict):
    """File data that doesn't fit into other multimodal blocks.

    This block is intended for files that are not images, audio, or plaintext. For
    example, it can be used for PDFs, Word documents, etc.

    If the file is an image, audio, or plaintext, you should use the corresponding
    content block type (e.g., ``ImageContentBlock``, ``AudioContentBlock``,
    ``PlainTextContentBlock``).

    .. note::
        ``create_file_block`` may also be used as a factory to create a
        ``FileContentBlock``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["file"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    file_id: NotRequired[str]
    """ID of the file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the file. Required for base64.

    `Examples from IANA <https://www.iana.org/assignments/media-types/media-types.xhtml>`__

    """

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the file."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


# Future modalities to consider:
# - 3D models
# - Tabular data


class NonStandardContentBlock(TypedDict):
    """Provider-specific data.

    This block contains data for which there is not yet a standard type.

    The purpose of this block should be to simply hold a provider-specific payload.
    If a provider's non-standard output includes reasoning and tool calls, it should be
    the adapter's job to parse that payload and emit the corresponding standard
    ``ReasoningContentBlock`` and ``ToolCallContentBlocks``.

    .. note::
        ``create_non_standard_block`` may also be used as a factory to create a
        ``NonStandardContentBlock``. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["non_standard"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier. Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (``UUID4`` prefixed with ``'lc_'``))
    """

    value: dict[str, Any]
    """Provider-specific data."""

    index: NotRequired[int]
    """Index of block in aggregate response. Used during streaming."""


# --- Aliases ---
DataContentBlock = Union[
    ImageContentBlock,
    VideoContentBlock,
    AudioContentBlock,
    PlainTextContentBlock,
    FileContentBlock,
]

ToolContentBlock = Union[
    ToolCall,
    ToolCallChunk,
    CodeInterpreterCall,
    CodeInterpreterOutput,
    CodeInterpreterResult,
    WebSearchCall,
    WebSearchResult,
]

ContentBlock = Union[
    TextContentBlock,
    ToolCall,
    ToolCallChunk,
    InvalidToolCall,
    ReasoningContentBlock,
    NonStandardContentBlock,
    DataContentBlock,
    ToolContentBlock,
]


KNOWN_BLOCK_TYPES = {
    "text",
    "text-plain",
    "tool_call",
    "invalid_tool_call",
    "tool_call_chunk",
    "reasoning",
    "non_standard",
    "image",
    "audio",
    "file",
    "video",
    "code_interpreter_call",
    "code_interpreter_output",
    "code_interpreter_result",
    "web_search_call",
    "web_search_result",
}


def is_data_content_block(block: dict) -> bool:
    """Check if the content block is a standard data content block.

    Args:
        block: The content block to check.

    Returns:
        True if the content block is a data content block, False otherwise.
    """
    return block.get("type") in (
        "audio",
        "image",
        "video",
        "file",
        "text-plain",
    ) and any(
        key in block
        for key in (
            "url",
            "base64",
            "file_id",
            "text",
            "source_type",  # backwards compatibility
        )
    )


def is_tool_call_block(block: ContentBlock) -> TypeGuard[ToolCall]:
    """Type guard to check if a content block is a ``ToolCall``."""
    return block.get("type") == "tool_call"


def is_tool_call_chunk(block: ContentBlock) -> TypeGuard[ToolCallChunk]:
    """Type guard to check if a content block is a ``ToolCallChunk``."""
    return block.get("type") == "tool_call_chunk"


def is_text_block(block: ContentBlock) -> TypeGuard[TextContentBlock]:
    """Type guard to check if a content block is a ``TextContentBlock``."""
    return block.get("type") == "text"


def is_reasoning_block(block: ContentBlock) -> TypeGuard[ReasoningContentBlock]:
    """Type guard to check if a content block is a ``ReasoningContentBlock``."""
    return block.get("type") == "reasoning"


def is_invalid_tool_call_block(
    block: ContentBlock,
) -> TypeGuard[InvalidToolCall]:
    """Type guard to check if a content block is an ``InvalidToolCall``."""
    return block.get("type") == "invalid_tool_call"


def convert_to_openai_image_block(block: dict[str, Any]) -> dict:
    """Convert image content block to format expected by OpenAI Chat Completions API."""
    if "url" in block:
        return {
            "type": "image_url",
            "image_url": {
                "url": block["url"],
            },
        }
    if "base64" in block or block.get("source_type") == "base64":
        if "mime_type" not in block:
            error_message = "mime_type key is required for base64 data."
            raise ValueError(error_message)
        mime_type = block["mime_type"]
        base64_data = block["data"] if "data" in block else block["base64"]
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_data}",
            },
        }
    error_message = "Unsupported source type. Only 'url' and 'base64' are supported."
    raise ValueError(error_message)


def convert_to_openai_data_block(block: dict) -> dict:
    """Format standard data content block to format expected by OpenAI."""
    if block["type"] == "image":
        formatted_block = convert_to_openai_image_block(block)

    elif block["type"] == "file":
        if "base64" in block or block.get("source_type") == "base64":
            base64_data = block["data"] if "source_type" in block else block["base64"]
            file = {"file_data": f"data:{block['mime_type']};base64,{base64_data}"}
            if filename := block.get("filename"):
                file["filename"] = filename
            elif (extras := block.get("extras")) and ("filename" in extras):
                file["filename"] = extras["filename"]
            elif (extras := block.get("metadata")) and ("filename" in extras):
                # Backward compat
                file["filename"] = extras["filename"]
            else:
                warnings.warn(
                    "OpenAI may require a filename for file inputs. Specify a filename "
                    "in the content block: {'type': 'file', 'mime_type': "
                    "'application/pdf', 'base64': '...', 'filename': 'my-pdf'}",
                    stacklevel=1,
                )
            formatted_block = {"type": "file", "file": file}
        elif "file_id" in block or block.get("source_type") == "id":
            file_id = block["id"] if "source_type" in block else block["file_id"]
            formatted_block = {"type": "file", "file": {"file_id": file_id}}
        else:
            error_msg = "Keys base64 or file_id required for file blocks."
            raise ValueError(error_msg)

    elif block["type"] == "audio":
        if "base64" in block or block.get("source_type") == "base64":
            base64_data = block["data"] if "source_type" in block else block["base64"]
            audio_format = block["mime_type"].split("/")[-1]
            formatted_block = {
                "type": "input_audio",
                "input_audio": {"data": base64_data, "format": audio_format},
            }
        else:
            error_msg = "Key base64 is required for audio blocks."
            raise ValueError(error_msg)
    else:
        error_msg = f"Block of type {block['type']} is not supported."
        raise ValueError(error_msg)

    return formatted_block


def create_text_block(
    text: str,
    *,
    id: Optional[str] = None,
    annotations: Optional[list[Annotation]] = None,
    index: Optional[int] = None,
) -> TextContentBlock:
    """Create a ``TextContentBlock``.

    Args:
        text: The text content of the block.
        id: Content block identifier. Generated automatically if not provided.
        annotations: ``Citation``s and other annotations for the text.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``TextContentBlock``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    block = TextContentBlock(
        type="text",
        text=text,
        id=_ensure_id(id),
    )
    if annotations is not None:
        block["annotations"] = annotations
    if index is not None:
        block["index"] = index
    return block


def create_image_block(
    *,
    url: Optional[str] = None,
    base64: Optional[str] = None,
    file_id: Optional[str] = None,
    mime_type: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> ImageContentBlock:
    """Create an ``ImageContentBlock``.

    Args:
        url: URL of the image.
        base64: Base64-encoded image data.
        file_id: ID of the image file from a file storage system.
        mime_type: MIME type of the image. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``ImageContentBlock``.

    Raises:
        ValueError: If no image source is provided or if ``base64`` is used without
            ``mime_type``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    block = ImageContentBlock(type="image", id=_ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    return block


def create_video_block(
    *,
    url: Optional[str] = None,
    base64: Optional[str] = None,
    file_id: Optional[str] = None,
    mime_type: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> VideoContentBlock:
    """Create a ``VideoContentBlock``.

    Args:
        url: URL of the video.
        base64: Base64-encoded video data.
        file_id: ID of the video file from a file storage system.
        mime_type: MIME type of the video. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``VideoContentBlock``.

    Raises:
        ValueError: If no video source is provided or if ``base64`` is used without
            ``mime_type``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = VideoContentBlock(type="video", id=_ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    return block


def create_audio_block(
    *,
    url: Optional[str] = None,
    base64: Optional[str] = None,
    file_id: Optional[str] = None,
    mime_type: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> AudioContentBlock:
    """Create an ``AudioContentBlock``.

    Args:
        url: URL of the audio.
        base64: Base64-encoded audio data.
        file_id: ID of the audio file from a file storage system.
        mime_type: MIME type of the audio. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``AudioContentBlock``.

    Raises:
        ValueError: If no audio source is provided or if ``base64`` is used without
            ``mime_type``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = AudioContentBlock(type="audio", id=_ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    return block


def create_file_block(
    *,
    url: Optional[str] = None,
    base64: Optional[str] = None,
    file_id: Optional[str] = None,
    mime_type: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> FileContentBlock:
    """Create a ``FileContentBlock``.

    Args:
        url: URL of the file.
        base64: Base64-encoded file data.
        file_id: ID of the file from a file storage system.
        mime_type: MIME type of the file. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``FileContentBlock``.

    Raises:
        ValueError: If no file source is provided or if ``base64`` is used without
            ``mime_type``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = FileContentBlock(type="file", id=_ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    return block


def create_plaintext_block(
    text: Optional[str] = None,
    url: Optional[str] = None,
    base64: Optional[str] = None,
    file_id: Optional[str] = None,
    title: Optional[str] = None,
    context: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> PlainTextContentBlock:
    """Create a ``PlainTextContentBlock``.

    Args:
        text: The plaintext content.
        url: URL of the plaintext file.
        base64: Base64-encoded plaintext data.
        file_id: ID of the plaintext file from a file storage system.
        title: Title of the text data.
        context: Context or description of the text content.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``PlainTextContentBlock``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    block = PlainTextContentBlock(
        type="text-plain",
        mime_type="text/plain",
        id=_ensure_id(id),
    )

    if text is not None:
        block["text"] = text
    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if title is not None:
        block["title"] = title
    if context is not None:
        block["context"] = context
    if index is not None:
        block["index"] = index

    return block


def create_tool_call(
    name: str,
    args: dict[str, Any],
    *,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> ToolCall:
    """Create a ``ToolCall``.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
        id: An identifier for the tool call. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``ToolCall``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    block = ToolCall(
        type="tool_call",
        name=name,
        args=args,
        id=_ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    return block


def create_reasoning_block(
    reasoning: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> ReasoningContentBlock:
    """Create a ``ReasoningContentBlock``.

    Args:
        reasoning: The reasoning text or thought summary.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``ReasoningContentBlock``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    block = ReasoningContentBlock(
        type="reasoning",
        reasoning=reasoning or "",
        id=_ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    return block


def create_citation(
    *,
    url: Optional[str] = None,
    title: Optional[str] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    cited_text: Optional[str] = None,
    id: Optional[str] = None,
) -> Citation:
    """Create a ``Citation``.

    Args:
        url: URL of the document source.
        title: Source document title.
        start_index: Start index in the response text where citation applies.
        end_index: End index in the response text where citation applies.
        cited_text: Excerpt of source text being cited.
        id: Content block identifier. Generated automatically if not provided.

    Returns:
        A properly formatted ``Citation``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    block = Citation(type="citation", id=_ensure_id(id))

    if url is not None:
        block["url"] = url
    if title is not None:
        block["title"] = title
    if start_index is not None:
        block["start_index"] = start_index
    if end_index is not None:
        block["end_index"] = end_index
    if cited_text is not None:
        block["cited_text"] = cited_text

    return block


def create_non_standard_block(
    value: dict[str, Any],
    *,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> NonStandardContentBlock:
    """Create a ``NonStandardContentBlock``.

    Args:
        value: Provider-specific data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted ``NonStandardContentBlock``.

    .. note::
        The ``id`` is generated automatically if not provided, using a UUID4 format
        prefixed with ``'lc_'`` to indicate it is a LangChain-generated ID.

    """
    block = NonStandardContentBlock(
        type="non_standard",
        value=value,
        id=_ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    return block
