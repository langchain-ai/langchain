from __future__ import annotations

import base64 as base64_lib
import re
import uuid
from typing import Any, List, Literal, Optional, Sequence, Union

from pydantic import model_validator
from typing_extensions import Annotated, TypedDict

from langchain_core.messages.base_content import (
    AudioContentBlock,
    ContentBlock,
    DataContentBlock,
    FileContentBlock,
    GuardrailsContentBlock,
    ImageContentBlock,
    ReasoningContentBlock,
    RedactedReasoningContentBlock,
    TextContentBlock,
    ThinkingContentBlock,
    VideoContentBlock,
)
from langchain_core.messages.base_content import (
    ContentBlockChunk as ContentBlockChunk,
)
from langchain_core.utils._merge import merge_dicts

__all__ = [
    "AudioContentBlock",
    "ContentBlock",
    "ContentBlockChunk",
    "DataContentBlock",
    "FileContentBlock",
    "GuardrailsContentBlock",
    "ImageContentBlock",
    "ReasoningContentBlock",
    "RedactedReasoningContentBlock",
    "TextContentBlock",
    "ThinkingContentBlock",
    "VideoContentBlock",
    "create_audio_block",
    "create_file_block",
    "create_image_block",
    "create_text_block",
    "create_video_block",
    "ensure_id",
    "merge_content",
]


def ensure_id(id: Optional[str]) -> str:
    """Ensure that the id is set.

    Args:
        id: The id to ensure.

    Returns:
        The id if it is set, otherwise a new UUID4 prefixed with 'lc_'.
    """
    if id is None:
        return f"lc_{uuid.uuid4().hex}"
    return id


def create_text_block(text: str) -> TextContentBlock:
    """Create a `TextContentBlock`.

    Args:
        text: The text content.

    Returns:
        A properly formatted `TextContentBlock`.
    """
    return TextContentBlock(type="text", text=text)


def _convert_to_base64(data: Union[bytes, str]) -> str:
    if isinstance(data, bytes):
        return base64_lib.b64encode(data).decode("utf-8")
    return data


def convert_to_openai_image_block(block: ImageContentBlock) -> dict:
    """Convert an `ImageContentBlock` to an OpenAI-compatible image block.

    Args:
        block: The image block to convert.

    Returns:
        An OpenAI-compatible image block dict.
    """
    if url := block.get("url"):
        return {"type": "image_url", "image_url": {"url": url}}
    elif base64_data := block.get("base64"):
        mime_type = block.get("mime_type", "image/jpeg")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
        }
    msg = "Image block must have either url or base64 data"
    raise ValueError(msg)


def convert_to_openai_data_block(block: DataContentBlock) -> dict:
    """Convert a `DataContentBlock` to an OpenAI-compatible data block.

    Args:
        block: The data block to convert.

    Returns:
        An OpenAI-compatible data block dict.
    """
    if file_id := block.get("file_id"):
        return {"type": "file", "file": {"file_id": file_id}}

    file_data: dict = {}
    if mime_type := block.get("mime_type"):
        file_data["mime_type"] = mime_type
    if filename := block.get("filename"):
        file_data["filename"] = filename
    if base64_data := block.get("base64"):
        file_data["file_data"] = f"data:{mime_type};base64,{base64_data}"
    return {"type": "file", "file": file_data}


def merge_content(
    first_content: Union[str, List[Union[str, dict]]],
    *contents: Union[str, List[Union[str, dict]]],
) -> Union[str, List[Union[str, dict]]]:
    """Merge two message contents.

    Args:
        first_content: The base message content.
        *contents: The additional message contents to merge into the first.

    Returns:
        The merged message content.
    """
    merged = first_content
    for content in contents:
        # If current is a string and so is the next one
        if isinstance(merged, str):
            # If the next piece is also a string, simply concatenate
            if isinstance(content, str):
                merged += content
            # If the next piece is a list, convert the current to a list and merge
            else:
                merged = [*_as_list(merged), *content]
        elif isinstance(merged, list):
            # If the next piece is a string, append it to the current list
            if isinstance(content, str):
                merged = [*merged, content]
            else:
                # Recursively merge the first and last element of the list
                # if they are both dicts
                if merged and content:
                    last = merged[-1]
                    first = content[0]
                    if isinstance(last, dict) and isinstance(first, dict):
                        merged = [*merged[:-1], merge_dicts(last, first), *content[1:]]
                    else:
                        merged = [*merged, *content]
                else:
                    merged = [*merged, *content]
    return merged


def _as_list(content: Union[str, List]) -> List:
    if isinstance(content, list):
        return content
    return [content]


class _ImageBlockData(TypedDict, total=False):
    """Data for an image block, with source priority."""

    source_type: Literal["url", "base64", "file"]
    url: str
    media_type: str
    data: str
    file_id: str


class _AudioBlockData(TypedDict, total=False):
    """Data for an audio block, with source priority."""

    source_type: Literal["url", "base64", "file"]
    url: str
    media_type: str
    data: str
    file_id: str


class _VideoBlockData(TypedDict, total=False):
    """Data for a video block, with source priority."""

    source_type: Literal["url", "base64", "file"]
    url: str
    media_type: str
    data: str
    file_id: str


class _DocumentBlockData(TypedDict, total=False):
    """Data for a document block, with source priority."""

    source_type: Literal["url", "base64", "file", "text"]
    url: str
    media_type: str
    data: str
    file_id: str
    text: str


def _convert_block_to_anthropic_image_block(block: ImageContentBlock) -> dict:
    """Convert an image content block to Anthropic's image block format."""
    source_data: _ImageBlockData = {}
    if url := block.get("url"):
        source_data["source_type"] = "url"
        source_data["url"] = url
    elif base64_data := block.get("base64"):
        source_data["source_type"] = "base64"
        source_data["media_type"] = block.get("mime_type", "image/jpeg")
        source_data["data"] = base64_data
    elif file_id := block.get("file_id"):
        source_data["source_type"] = "file"
        source_data["file_id"] = file_id

    return {
        "type": "image",
        "source": source_data,
    }


def _convert_block_to_anthropic_document_block(block: FileContentBlock) -> dict:
    """Convert a file content block to Anthropic's document block format."""
    source_data: _DocumentBlockData = {}
    if url := block.get("url"):
        source_data["source_type"] = "url"
        source_data["url"] = url
    elif base64_data := block.get("base64"):
        source_data["source_type"] = "base64"
        source_data["media_type"] = block.get("mime_type", "application/pdf")
        source_data["data"] = base64_data
    elif file_id := block.get("file_id"):
        source_data["source_type"] = "file"
        source_data["file_id"] = file_id

    result: dict = {
        "type": "document",
        "source": source_data,
    }
    if title := block.get("title"):
        result["title"] = title
    if context := block.get("context"):
        result["context"] = context
    if citations := block.get("citations"):
        result["citations"] = citations

    return result


def _convert_block_to_anthropic_audio_block(block: AudioContentBlock) -> dict:
    """Convert an audio content block to Anthropic's audio block format."""
    source_data: _AudioBlockData = {}
    if url := block.get("url"):
        source_data["source_type"] = "url"
        source_data["url"] = url
    elif base64_data := block.get("base64"):
        source_data["source_type"] = "base64"
        source_data["media_type"] = block.get("mime_type", "audio/wav")
        source_data["data"] = base64_data
    elif file_id := block.get("file_id"):
        source_data["source_type"] = "file"
        source_data["file_id"] = file_id

    return {
        "type": "audio",
        "source": source_data,
    }


def _convert_block_to_anthropic_video_block(block: VideoContentBlock) -> dict:
    """Convert a video content block to Anthropic's video block format."""
    source_data: _VideoBlockData = {}
    if url := block.get("url"):
        source_data["source_type"] = "url"
        source_data["url"] = url
    elif base64_data := block.get("base64"):
        source_data["source_type"] = "base64"
        source_data["media_type"] = block.get("mime_type", "video/mp4")
        source_data["data"] = base64_data
    elif file_id := block.get("file_id"):
        source_data["source_type"] = "file"
        source_data["file_id"] = file_id

    return {
        "type": "video",
        "source": source_data,
    }


def _extract_token_usage(
    response: dict,
) -> dict:
    """Extract token usage information from a response.

    Args:
        response: The response from which to extract token usage.

    Returns:
        A dictionary with token usage information.
    """
    usage: dict = {}
    if input_tokens := response.get("usage", {}).get("input_tokens"):
        usage["input_tokens"] = input_tokens
    if output_tokens := response.get("usage", {}).get("output_tokens"):
        usage["output_tokens"] = output_tokens
    return usage


def _url_type(url: str) -> Literal["url", "base64"]:
    """Determine if a URL is a base64 data URL or a regular URL.

    Args:
        url: The URL to check.

    Returns:
        "base64" if the URL is a base64 data URL, "url" otherwise.
    """
    if url.startswith("data:"):
        return "base64"
    return "url"


def _parse_data_url(url: str) -> tuple[str, str]:
    """Parse a base64 data URL into its mime type and data.

    Args:
        url: The base64 data URL to parse.

    Returns:
        A tuple of (mime_type, base64_data).
    """
    # data:[<mediatype>][;base64],<data>
    match = re.match(r"data:([^;,]+)(?:;[^,]+)?,(.+)", url, re.DOTALL)
    if not match:
        msg = f"Invalid data URL: {url[:50]}"
        raise ValueError(msg)
    return match.group(1), match.group(2)


def _convert_v0_image_block(block: dict) -> ImageContentBlock:
    """Convert a v0-style image block to an ImageContentBlock.

    Args:
        block: A v0-style image block dict.

    Returns:
        An ImageContentBlock.
    """
    if image_url := block.get("image_url"):
        url = image_url if isinstance(image_url, str) else image_url.get("url", "")
        if _url_type(url) == "base64":
            mime_type, data = _parse_data_url(url)
            return ImageContentBlock(
                type="image",
                id=ensure_id(None),
                base64=data,
                mime_type=mime_type,
            )
        return ImageContentBlock(type="image", id=ensure_id(None), url=url)
    msg = "Image block must have image_url"
    raise ValueError(msg)


def _convert_v0_audio_block(block: dict) -> AudioContentBlock:
    """Convert a v0-style audio block to an AudioContentBlock.

    Args:
        block: A v0-style audio block dict.

    Returns:
        An AudioContentBlock.
    """
    if audio_url := block.get("audio_url"):
        url = audio_url if isinstance(audio_url, str) else audio_url.get("url", "")
        if _url_type(url) == "base64":
            mime_type, data = _parse_data_url(url)
            return AudioContentBlock(
                type="audio",
                id=ensure_id(None),
                base64=data,
                mime_type=mime_type,
            )
        return AudioContentBlock(type="audio", id=ensure_id(None), url=url)
    elif input_audio := block.get("input_audio"):
        data = input_audio.get("data", "")
        format_ = input_audio.get("format", "wav")
        mime_type = f"audio/{format_}"
        return AudioContentBlock(
            type="audio",
            id=ensure_id(None),
            base64=data,
            mime_type=mime_type,
        )
    msg = "Audio block must have audio_url or input_audio"
    raise ValueError(msg)


def _convert_v0_video_block(block: dict) -> VideoContentBlock:
    """Convert a v0-style video block to a VideoContentBlock.

    Args:
        block: A v0-style video block dict.

    Returns:
        A VideoContentBlock.
    """
    if video_url := block.get("video_url"):
        url = video_url if isinstance(video_url, str) else video_url.get("url", "")
        if _url_type(url) == "base64":
            mime_type, data = _parse_data_url(url)
            return VideoContentBlock(
                type="video",
                id=ensure_id(None),
                base64=data,
                mime_type=mime_type,
            )
        return VideoContentBlock(type="video", id=ensure_id(None), url=url)
    msg = "Video block must have video_url"
    raise ValueError(msg)


def _convert_v0_file_block(block: dict) -> FileContentBlock:
    """Convert a v0-style file block to a FileContentBlock.

    Args:
        block: A v0-style file block dict.

    Returns:
        A FileContentBlock.
    """
    if file_url := block.get("file_url"):
        url = file_url if isinstance(file_url, str) else file_url.get("url", "")
        if _url_type(url) == "base64":
            mime_type, data = _parse_data_url(url)
            return FileContentBlock(
                type="file",
                id=ensure_id(None),
                base64=data,
                mime_type=mime_type,
            )
        return FileContentBlock(type="file", id=ensure_id(None), url=url)
    msg = "File block must have file_url"
    raise ValueError(msg)


def convert_to_blocks(
    content: Union[str, List[Union[str, dict]]],
) -> List[ContentBlock]:
    """Convert message content into a list of ContentBlocks.

    Handles v0-style blocks (``{"type": "image_url", ...}``) as well as v1
    blocks that are already typed ContentBlock dicts.

    Args:
        content: Message content — either a plain string or a list of string /
            block-dict items.

    Returns:
        A list of typed ``ContentBlock`` items.
    """
    if isinstance(content, str):
        return [create_text_block(content)]
    blocks: List[ContentBlock] = []
    for item in content:
        if isinstance(item, str):
            blocks.append(create_text_block(item))
        elif isinstance(item, dict):
            type_ = item.get("type")
            if type_ == "text":
                blocks.append(TextContentBlock(**item))
            elif type_ == "image":
                blocks.append(ImageContentBlock(**item))  # type: ignore[arg-type]
            elif type_ == "audio":
                blocks.append(AudioContentBlock(**item))  # type: ignore[arg-type]
            elif type_ == "video":
                blocks.append(VideoContentBlock(**item))  # type: ignore[arg-type]
            elif type_ == "file":
                blocks.append(FileContentBlock(**item))  # type: ignore[arg-type]
            elif type_ == "image_url":
                blocks.append(_convert_v0_image_block(item))
            elif type_ == "audio_url" or type_ == "input_audio":
                blocks.append(_convert_v0_audio_block(item))
            elif type_ == "video_url":
                blocks.append(_convert_v0_video_block(item))
            elif type_ == "file_url":
                blocks.append(_convert_v0_file_block(item))
            else:
                blocks.append(item)  # type: ignore[arg-type]
        else:
            blocks.append(item)  # type: ignore[arg-type]
    return blocks


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
        mime_type: MIME type of the image.

            Required for base64 data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

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

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
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
        mime_type: MIME type of the video.

            Required for base64 data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

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
        mime_type: MIME type of the audio.

            Required for base64 data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

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
    title: str | None = None,
    context: str | None = None,
    citations: dict | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> FileContentBlock:
    """Create a `FileContentBlock`.

    Args:
        url: URL of the file.
        base64: Base64-encoded file data.
        file_id: ID of the file from a file storage system.
        mime_type: MIME type of the file.

            Required for base64 data.
        title: Title of the file.
        context: Context for the file.
        citations: Citations for the file.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

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
    if title is not None:
        block["title"] = title
    if context is not None:
        block["context"] = context
    if citations is not None:
        block["citations"] = citations
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block
