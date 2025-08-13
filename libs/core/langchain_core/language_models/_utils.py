import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
from langchain_core.messages.content_blocks import (
    ContentBlock,
    create_audio_block,
    create_file_block,
    create_non_standard_block,
)


def _is_openai_data_block(block: dict) -> bool:
    """Check if the block contains multimodal data in OpenAI Chat Completions format."""
    if block.get("type") == "image_url":
        if (
            (set(block.keys()) <= {"type", "image_url", "detail"})
            and (image_url := block.get("image_url"))
            and isinstance(image_url, dict)
        ):
            url = image_url.get("url")
            if isinstance(url, str):
                return True

    elif block.get("type") == "file":
        if (file := block.get("file")) and isinstance(file, dict):
            file_data = file.get("file_data")
            if isinstance(file_data, str):
                return True

    elif block.get("type") == "input_audio":
        if (input_audio := block.get("input_audio")) and isinstance(input_audio, dict):
            audio_data = input_audio.get("data")
            audio_format = input_audio.get("format")
            if isinstance(audio_data, str) and isinstance(audio_format, str):
                return True

    else:
        return False

    return False


def _parse_data_uri(uri: str) -> Optional[dict]:
    """Parse a data URI into its components. If parsing fails, return None.

    Example:

        .. code-block:: python

            data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            parsed = _parse_data_uri(data_uri)

            assert parsed == {
                "source_type": "base64",
                "mime_type": "image/jpeg",
                "data": "/9j/4AAQSkZJRg...",
            }

    """
    regex = r"^data:(?P<mime_type>[^;]+);base64,(?P<data>.+)$"
    match = re.match(regex, uri)
    if match is None:
        return None
    return {
        "source_type": "base64",
        "data": match.group("data"),
        "mime_type": match.group("mime_type"),
    }


def _convert_openai_format_to_data_block(block: dict) -> ContentBlock:
    """Convert OpenAI image content block to standard data content block.

    If parsing fails, pass-through.

    Args:
        block: The OpenAI image content block to convert.

    Returns:
        The converted standard data content block.
    """
    if block["type"] == "input_audio":
        data = block["input_audio"].get("data")
        audio_format = block["input_audio"].get("format")
        if data and audio_format:
            return create_audio_block(
                base64=data,
                mime_type=f"audio/{audio_format}",
            )

    if block["type"] == "file":
        parsed = _parse_data_uri(block["file"]["file_data"])
        if parsed is not None:
            base64_data = parsed.pop("data")
            mime_type = parsed.pop("mime_type")
            filename = block["file"].get("filename")
            return create_file_block(
                base64=base64_data,
                mime_type=mime_type,
                filename=filename,
            )

    # This logic is and never was used but @ccurme wanted to keep it
    # if block["type"] == "image_url":
    #     parsed = _parse_data_uri(block["image_url"]["url"])
    #     if parsed is not None:
    #         base64 = parsed.pop("data", None)
    #         mime_type = parsed.pop("mime_type", None)
    #         if base64 and mime_type:
    #             return create_image_block(
    #                 base64=base64,
    #                 mime_type=mime_type,
    #             )

    return create_non_standard_block(
        value=block,
    )


def _normalize_messages(messages: Sequence["BaseMessage"]) -> list["BaseMessage"]:
    """Extend support for message formats.

    Chat models implement support for images in OpenAI Chat Completions format, as well
    as other multimodal data as standard data blocks. This function extends support to
    audio and file data in OpenAI Chat Completions format by converting them to standard
    data blocks.
    """
    formatted_messages = []
    for message in messages:
        formatted_message = message
        if isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                if (
                    isinstance(block, dict)
                    # Subset to (PDF) files and audio, as most relevant chat models
                    # support images in OAI format (and some may not yet support the
                    # standard data block format)
                    and block.get("type") in {"file", "input_audio"}
                    and _is_openai_data_block(block)
                ):
                    if formatted_message is message:
                        formatted_message = message.model_copy()
                        # Also shallow-copy content
                        formatted_message.content = list(formatted_message.content)

                    formatted_message.content[idx] = (  # type: ignore[call-overload,index]  # mypy confused by .model_copy
                        _convert_openai_format_to_data_block(block)
                    )
        formatted_messages.append(formatted_message)

    return formatted_messages
