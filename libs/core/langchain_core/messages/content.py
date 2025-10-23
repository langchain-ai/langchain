"""Standard, multimodal content blocks for Large Language Model I/O.

!!! warning
    This module is under active development. The API is unstable and subject to
    change in future releases.

This module provides standardized data structures for representing inputs to and
outputs from LLMs. The core abstraction is the **Content Block**, a `TypedDict`.

**Rationale**

Different LLM providers use distinct and incompatible API schemas. This module
provides a unified, provider-agnostic format to facilitate these interactions. A
message to or from a model is simply a list of content blocks, allowing for the natural
interleaving of text, images, and other content in a single ordered sequence.

An adapter for a specific provider is responsible for translating this standard list of
blocks into the format required by its API.

**Extensibility**

Data **not yet mapped** to a standard block may be represented using the
`NonStandardContentBlock`, which allows for provider-specific data to be included
without losing the benefits of type checking and validation.

Furthermore, provider-specific fields **within** a standard block are fully supported
by default in the `extras` field of each block. This allows for additional metadata
to be included without breaking the standard structure.

!!! warning
    Do not heavily rely on the `extras` field for provider-specific data! This field
    is subject to deprecation in future releases as we move towards PEP 728.

!!! note
    Following widespread adoption of [PEP 728](https://peps.python.org/pep-0728/), we
    will add `extra_items=Any` as a param to Content Blocks. This will signify to type
    checkers that additional provider-specific fields are allowed outside of the
    `extras` field, and that will become the new standard approach to adding
    provider-specific metadata.

    ??? note

        **Example with PEP 728 provider-specific fields:**

        ```python
        # Content block definition
        # NOTE: `extra_items=Any`
        class TextContentBlock(TypedDict, extra_items=Any):
            type: Literal["text"]
            id: NotRequired[str]
            text: str
            annotations: NotRequired[list[Annotation]]
            index: NotRequired[int]
        ```

        ```python
        from langchain_core.messages.content import TextContentBlock

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
        ```

        PEP 728 is enabled with `# type: ignore[call-arg]` comments to suppress
        warnings from type checkers that don't yet support it. The functionality works
        correctly in Python 3.13+ and will be fully supported as the ecosystem catches
        up.

**Key Block Types**

The module defines several types of content blocks, including:

- `TextContentBlock`: Standard text output.
- `Citation`: For annotations that link text output to a source document.
- `ToolCall`: For function calling.
- `ReasoningContentBlock`: To capture a model's thought process.
- Multimodal data:
    - `ImageContentBlock`
    - `AudioContentBlock`
    - `VideoContentBlock`
    - `PlainTextContentBlock` (e.g. .txt or .md files)
    - `FileContentBlock` (e.g. PDFs, etc.)

**Example Usage**

```python
# Direct construction:
from langchain_core.messages.content import TextContentBlock, ImageContentBlock

multimodal_message: AIMessage(
    content_blocks=[
        TextContentBlock(type="text", text="What is shown in this image?"),
        ImageContentBlock(
            type="image",
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]
)

# Using factories:
from langchain_core.messages.content import create_text_block, create_image_block

multimodal_message: AIMessage(
    content=[
        create_text_block("What is shown in this image?"),
        create_image_block(
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]
)
```

Factory functions offer benefits such as:
- Automatic ID generation (when not provided)
- No need to manually specify the `type` field
"""

from typing import Any, Literal, get_args, get_type_hints

from typing_extensions import NotRequired, TypedDict

from langchain_core.utils.utils import ensure_id


class Citation(TypedDict):
    """Annotation for citing data from a document.

    !!! note
        `start`/`end` indices refer to the **response text**,
        not the source text. This means that the indices are relative to the model's
        response, not the original document (as specified in the `url`).

    !!! note "Factory function"
        `create_citation` may also be used as a factory to create a `Citation`.
        Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["citation"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    url: NotRequired[str]
    """URL of the document source."""

    title: NotRequired[str]
    """Source document title.

    For example, the page title for a web page or the title of a paper.
    """

    start_index: NotRequired[int]
    """Start index of the **response text** (`TextContentBlock.text`)."""

    end_index: NotRequired[int]
    """End index of the **response text** (`TextContentBlock.text`)"""

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
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    value: dict[str, Any]
    """Provider-specific annotation data."""


Annotation = Citation | NonStandardAnnotation
"""A union of all defined `Annotation` types."""


class TextContentBlock(TypedDict):
    """Text output from a LLM.

    This typically represents the main text content of a message, such as the response
    from a language model or the text of a user message.

    !!! note "Factory function"
        `create_text_block` may also be used as a factory to create a
        `TextContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["text"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    text: str
    """Block text."""

    annotations: NotRequired[list[Annotation]]
    """`Citation`s and other annotations."""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ToolCall(TypedDict):
    """Represents an AI's request to call a tool.

    Example:
        ```python
        {"name": "foo", "args": {"a": 1}, "id": "123"}
        ```

        This represents a request to call the tool named "foo" with arguments {"a": 1}
        and an identifier of "123".

    !!! note "Factory function"
        `create_tool_call` may also be used as a factory to create a
        `ToolCall`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["tool_call"]
    """Used for discrimination."""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.

    """
    # TODO: Consider making this NotRequired[str] in the future.

    name: str
    """The name of the tool to be called."""

    args: dict[str, Any]
    """The arguments to the tool call."""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ToolCallChunk(TypedDict):
    """A chunk of a tool call (yielded when streaming).

    When merging `ToolCallChunks` (e.g., via `AIMessageChunk.__add__`),
    all string attributes are concatenated. Chunks are only merged if their
    values of `index` are equal and not `None`.

    Example:
    ```python
    left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
    right_chunks = [ToolCallChunk(name=None, args="1}", index=0)]

    (
        AIMessageChunk(content="", tool_call_chunks=left_chunks)
        + AIMessageChunk(content="", tool_call_chunks=right_chunks)
    ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]
    ```
    """

    # TODO: Consider making fields NotRequired[str] in the future.

    type: Literal["tool_call_chunk"]
    """Used for serialization."""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.

    """

    name: str | None
    """The name of the tool to be called."""

    args: str | None
    """The arguments to the tool call."""

    index: NotRequired[int | str]
    """The index of the tool call in a sequence."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class InvalidToolCall(TypedDict):
    """Allowance for errors made by LLM.

    Here we add an `error` key to surface errors made during generation
    (e.g., invalid JSON arguments.)

    """

    # TODO: Consider making fields NotRequired[str] in the future.

    type: Literal["invalid_tool_call"]
    """Used for discrimination."""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.

    """

    name: str | None
    """The name of the tool to be called."""

    args: str | None
    """The arguments to the tool call."""

    error: str | None
    """An error message associated with the tool call."""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ServerToolCall(TypedDict):
    """Tool call that is executed server-side.

    For example: code execution, web search, etc.
    """

    type: Literal["server_tool_call"]
    """Used for discrimination."""

    id: str
    """An identifier associated with the tool call."""

    name: str
    """The name of the tool to be called."""

    args: dict[str, Any]
    """The arguments to the tool call."""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ServerToolCallChunk(TypedDict):
    """A chunk of a server-side tool call (yielded when streaming)."""

    type: Literal["server_tool_call_chunk"]
    """Used for discrimination."""

    name: NotRequired[str]
    """The name of the tool to be called."""

    args: NotRequired[str]
    """JSON substring of the arguments to the tool call."""

    id: NotRequired[str]
    """An identifier associated with the tool call."""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ServerToolResult(TypedDict):
    """Result of a server-side tool call."""

    type: Literal["server_tool_result"]
    """Used for discrimination."""

    id: NotRequired[str]
    """An identifier associated with the server tool result."""

    tool_call_id: str
    """ID of the corresponding server tool call."""

    status: Literal["success", "error"]
    """Execution status of the server-side tool."""

    output: NotRequired[Any]
    """Output of the executed tool."""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


class ReasoningContentBlock(TypedDict):
    """Reasoning output from a LLM.

    !!! note "Factory function"
        `create_reasoning_block` may also be used as a factory to create a
        `ReasoningContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["reasoning"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    reasoning: NotRequired[str]
    """Reasoning text.

    Either the thought summary or the raw reasoning text itself. This is often parsed
    from `<think>` tags in the model's response.

    """

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata."""


# Note: `title` and `context` are fields that could be used to provide additional
# information about the file, such as a description or summary of its content.
# E.g. with Claude, you can provide a context for a file which is passed to the model.
class ImageContentBlock(TypedDict):
    """Image data.

    !!! note "Factory function"
        `create_image_block` may also be used as a factory to create a
        `ImageContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["image"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    file_id: NotRequired[str]
    """ID of the image file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the image. Required for base64.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml#image)

    """

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the image."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the image data itself."""


class VideoContentBlock(TypedDict):
    """Video data.

    !!! note "Factory function"
        `create_video_block` may also be used as a factory to create a
        `VideoContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["video"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    file_id: NotRequired[str]
    """ID of the video file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the video. Required for base64.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml#video)

    """

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the video."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the video data itself."""


class AudioContentBlock(TypedDict):
    """Audio data.

    !!! note "Factory function"
        `create_audio_block` may also be used as a factory to create an
        `AudioContentBlock`. Benefits include:
        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["audio"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    file_id: NotRequired[str]
    """ID of the audio file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the audio. Required for base64.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml#audio)

    """

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the audio."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the audio data itself."""


class PlainTextContentBlock(TypedDict):
    """Plaintext data (e.g., from a document).

    !!! note
        A `PlainTextContentBlock` existed in `langchain-core<1.0.0`. Although the
        name has carried over, the structure has changed significantly. The only shared
        keys between the old and new versions are `type` and `text`, though the
        `type` value has changed from `'text'` to `'text-plain'`.

    !!! note
        Title and context are optional fields that may be passed to the model. See
        Anthropic [example](https://docs.claude.com/en/docs/build-with-claude/citations#citable-vs-non-citable-content).

    !!! note "Factory function"
        `create_plaintext_block` may also be used as a factory to create a
        `PlainTextContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["text-plain"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    file_id: NotRequired[str]
    """ID of the plaintext file, e.g., from a file storage system."""

    mime_type: Literal["text/plain"]
    """MIME type of the file. Required for base64."""

    index: NotRequired[int | str]
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
    """Provider-specific metadata. This shouldn't be used for the data itself."""


class FileContentBlock(TypedDict):
    """File data that doesn't fit into other multimodal block types.

    This block is intended for files that are not images, audio, or plaintext. For
    example, it can be used for PDFs, Word documents, etc.

    If the file is an image, audio, or plaintext, you should use the corresponding
    content block type (e.g., `ImageContentBlock`, `AudioContentBlock`,
    `PlainTextContentBlock`).

    !!! note "Factory function"
        `create_file_block` may also be used as a factory to create a
        `FileContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["file"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    file_id: NotRequired[str]
    """ID of the file, e.g., from a file storage system."""

    mime_type: NotRequired[str]
    """MIME type of the file. Required for base64.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml)

    """

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""

    url: NotRequired[str]
    """URL of the file."""

    base64: NotRequired[str]
    """Data as a base64 string."""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the file data itself."""


# Future modalities to consider:
# - 3D models
# - Tabular data


class NonStandardContentBlock(TypedDict):
    """Provider-specific data.

    This block contains data for which there is not yet a standard type.

    The purpose of this block should be to simply hold a provider-specific payload.
    If a provider's non-standard output includes reasoning and tool calls, it should be
    the adapter's job to parse that payload and emit the corresponding standard
    `ReasoningContentBlock` and `ToolCalls`.

    Has no `extras` field, as provider-specific data should be included in the
    `value` field.

    !!! note "Factory function"
        `create_non_standard_block` may also be used as a factory to create a
        `NonStandardContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time

    """

    type: Literal["non_standard"]
    """Type of the content block. Used for discrimination."""

    id: NotRequired[str]
    """Content block identifier.

    Either:

    - Generated by the provider (e.g., OpenAI's file ID)
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))

    """

    value: dict[str, Any]
    """Provider-specific data."""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming."""


# --- Aliases ---
DataContentBlock = (
    ImageContentBlock
    | VideoContentBlock
    | AudioContentBlock
    | PlainTextContentBlock
    | FileContentBlock
)
"""A union of all defined multimodal data `ContentBlock` types."""

ToolContentBlock = (
    ToolCall | ToolCallChunk | ServerToolCall | ServerToolCallChunk | ServerToolResult
)

ContentBlock = (
    TextContentBlock
    | InvalidToolCall
    | ReasoningContentBlock
    | NonStandardContentBlock
    | DataContentBlock
    | ToolContentBlock
)
"""A union of all defined `ContentBlock` types and aliases."""


KNOWN_BLOCK_TYPES = {
    # Text output
    "text",
    "reasoning",
    # Tools
    "tool_call",
    "invalid_tool_call",
    "tool_call_chunk",
    # Multimodal data
    "image",
    "audio",
    "file",
    "text-plain",
    "video",
    # Server-side tool calls
    "server_tool_call",
    "server_tool_call_chunk",
    "server_tool_result",
    # Catch-all
    "non_standard",
    # citation and non_standard_annotation intentionally omitted
}
"""These are block types known to `langchain-core>=1.0.0`.

If a block has a type not in this set, it is considered to be provider-specific.
"""


def _get_data_content_block_types() -> tuple[str, ...]:
    """Get type literals from DataContentBlock union members dynamically.

    Example: ("image", "video", "audio", "text-plain", "file")

    Note that old style multimodal blocks type literals with new style blocks.
    Speficially, "image", "audio", and "file".

    See the docstring of `_normalize_messages` in `language_models._utils` for details.
    """
    data_block_types = []

    for block_type in get_args(DataContentBlock):
        hints = get_type_hints(block_type)
        if "type" in hints:
            type_annotation = hints["type"]
            if hasattr(type_annotation, "__args__"):
                # This is a Literal type, get the literal value
                literal_value = type_annotation.__args__[0]
                data_block_types.append(literal_value)

    return tuple(data_block_types)


def is_data_content_block(block: dict) -> bool:
    """Check if the provided content block is a data content block.

    Returns True for both v0 (old-style) and v1 (new-style) multimodal data blocks.

    Args:
        block: The content block to check.

    Returns:
        `True` if the content block is a data content block, `False` otherwise.

    """
    if block.get("type") not in _get_data_content_block_types():
        return False

    if any(key in block for key in ("url", "base64", "file_id", "text")):
        # Type is valid and at least one data field is present
        # (Accepts old-style image and audio URLContentBlock)

        # 'text' is checked to support v0 PlainTextContentBlock types
        # We must guard against new style TextContentBlock which also has 'text' `type`
        # by ensuring the presense of `source_type`
        if block["type"] == "text" and "source_type" not in block:  # noqa: SIM103  # This is more readable
            return False

        return True

    if "source_type" in block:
        # Old-style content blocks had possible types of 'image', 'audio', and 'file'
        # which is not captured in the prior check
        source_type = block["source_type"]
        if (source_type == "url" and "url" in block) or (
            source_type == "base64" and "data" in block
        ):
            return True
        if (source_type == "id" and "id" in block) or (
            source_type == "text" and "url" in block
        ):
            return True

    return False


def create_text_block(
    text: str,
    *,
    id: str | None = None,
    annotations: list[Annotation] | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> TextContentBlock:
    """Create a `TextContentBlock`.

    Args:
        text: The text content of the block.
        id: Content block identifier. Generated automatically if not provided.
        annotations: `Citation`s and other annotations for the text.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `TextContentBlock`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    block = TextContentBlock(
        type="text",
        text=text,
        id=ensure_id(id),
    )
    if annotations is not None:
        block["annotations"] = annotations
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_image_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ImageContentBlock:
    """Create an `ImageContentBlock`.

    Args:
        url: URL of the image.
        base64: Base64-encoded image data.
        file_id: ID of the image file from a file storage system.
        mime_type: MIME type of the image. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `ImageContentBlock`.

    Raises:
        ValueError: If no image source is provided or if `base64` is used without
            `mime_type`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    block = ImageContentBlock(type="image", id=ensure_id(id))

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

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_video_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> VideoContentBlock:
    """Create a `VideoContentBlock`.

    Args:
        url: URL of the video.
        base64: Base64-encoded video data.
        file_id: ID of the video file from a file storage system.
        mime_type: MIME type of the video. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `VideoContentBlock`.

    Raises:
        ValueError: If no video source is provided or if `base64` is used without
            `mime_type`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = VideoContentBlock(type="video", id=ensure_id(id))

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

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_audio_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> AudioContentBlock:
    """Create an `AudioContentBlock`.

    Args:
        url: URL of the audio.
        base64: Base64-encoded audio data.
        file_id: ID of the audio file from a file storage system.
        mime_type: MIME type of the audio. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `AudioContentBlock`.

    Raises:
        ValueError: If no audio source is provided or if `base64` is used without
            `mime_type`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = AudioContentBlock(type="audio", id=ensure_id(id))

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

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_file_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> FileContentBlock:
    """Create a `FileContentBlock`.

    Args:
        url: URL of the file.
        base64: Base64-encoded file data.
        file_id: ID of the file from a file storage system.
        mime_type: MIME type of the file. Required for base64 data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `FileContentBlock`.

    Raises:
        ValueError: If no file source is provided or if `base64` is used without
            `mime_type`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = FileContentBlock(type="file", id=ensure_id(id))

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

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_plaintext_block(
    text: str | None = None,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    title: str | None = None,
    context: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> PlainTextContentBlock:
    """Create a `PlainTextContentBlock`.

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
        A properly formatted `PlainTextContentBlock`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    block = PlainTextContentBlock(
        type="text-plain",
        mime_type="text/plain",
        id=ensure_id(id),
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

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_tool_call(
    name: str,
    args: dict[str, Any],
    *,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ToolCall:
    """Create a `ToolCall`.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
        id: An identifier for the tool call. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `ToolCall`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    block = ToolCall(
        type="tool_call",
        name=name,
        args=args,
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_reasoning_block(
    reasoning: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ReasoningContentBlock:
    """Create a `ReasoningContentBlock`.

    Args:
        reasoning: The reasoning text or thought summary.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `ReasoningContentBlock`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    block = ReasoningContentBlock(
        type="reasoning",
        reasoning=reasoning or "",
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_citation(
    *,
    url: str | None = None,
    title: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    cited_text: str | None = None,
    id: str | None = None,
    **kwargs: Any,
) -> Citation:
    """Create a `Citation`.

    Args:
        url: URL of the document source.
        title: Source document title.
        start_index: Start index in the response text where citation applies.
        end_index: End index in the response text where citation applies.
        cited_text: Excerpt of source text being cited.
        id: Content block identifier. Generated automatically if not provided.

    Returns:
        A properly formatted `Citation`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    block = Citation(type="citation", id=ensure_id(id))

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

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_non_standard_block(
    value: dict[str, Any],
    *,
    id: str | None = None,
    index: int | str | None = None,
) -> NonStandardContentBlock:
    """Create a `NonStandardContentBlock`.

    Args:
        value: Provider-specific data.
        id: Content block identifier. Generated automatically if not provided.
        index: Index of block in aggregate response. Used during streaming.

    Returns:
        A properly formatted `NonStandardContentBlock`.

    !!! note
        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.

    """
    block = NonStandardContentBlock(
        type="non_standard",
        value=value,
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    return block
