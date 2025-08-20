import re
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
from langchain_core.messages.content import (
    KNOWN_BLOCK_TYPES,
    ContentBlock,
    create_audio_block,
    create_file_block,
    create_image_block,
    create_plaintext_block,
)


def _is_openai_data_block(block: dict) -> bool:
    """Check if the block contains multimodal data in OpenAI Chat Completions format.

    Supports both data and ID-style blocks (e.g. ``'file_data'`` and ``'file_id'``).

    If additional keys are present, they are ignored / will not affect outcome as long
    as the required keys are present and valid.

    """
    if block.get("type") == "image_url":
        if (
            (set(block.keys()) <= {"type", "image_url", "detail"})
            and (image_url := block.get("image_url"))
            and isinstance(image_url, dict)
        ):
            url = image_url.get("url")
            if isinstance(url, str):
                # Required per OpenAI spec
                return True
            # Ignore `'detail'` since it's optional and specific to OpenAI

    elif block.get("type") == "input_audio":
        if (audio := block.get("input_audio")) and isinstance(audio, dict):
            audio_data = audio.get("data")
            audio_format = audio.get("format")
            # Both required per OpenAI spec
            if isinstance(audio_data, str) and isinstance(audio_format, str):
                return True

    elif block.get("type") == "file":
        if (file := block.get("file")) and isinstance(file, dict):
            file_data = file.get("file_data")
            file_id = file.get("file_id")
            # Files can be either base64-encoded or pre-uploaded with an ID
            if isinstance(file_data, str) or isinstance(file_id, str):
                return True

    else:
        return False

    # Has no `'type'` key
    return False


class ParsedDataUri(TypedDict):
    source_type: Literal["base64"]
    data: str
    mime_type: str


def _parse_data_uri(uri: str) -> Optional[ParsedDataUri]:
    """Parse a data URI into its components.

    If parsing fails, return None. If either MIME type or data is missing, return None.

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

    mime_type = match.group("mime_type")
    data = match.group("data")
    if not mime_type or not data:
        return None

    return {
        "source_type": "base64",
        "data": data,
        "mime_type": mime_type,
    }


def _convert_openai_format_to_data_block(
    block: dict,
) -> Union[ContentBlock, dict[Any, Any]]:
    """Convert OpenAI image/audio/file content block to respective v1 multimodal block.

    We expect that the incoming block is verified to be in OpenAI Chat Completions
    format.

    If parsing fails, passes block through unchanged.

    Mappings (Chat Completions to LangChain v1):
    - Image -> `ImageContentBlock`
    - Audio -> `AudioContentBlock`
    - File -> `FileContentBlock`

    """

    # Extract extra keys to put them in `extras`
    def _extract_extras(block_dict: dict, known_keys: set[str]) -> dict[str, Any]:
        """Extract unknown keys from block to preserve as extras."""
        return {k: v for k, v in block_dict.items() if k not in known_keys}

    # base64-style image block
    if (block["type"] == "image_url") and (
        parsed := _parse_data_uri(block["image_url"]["url"])
    ):
        known_keys = {"type", "image_url"}
        extras = _extract_extras(block, known_keys)

        # Also extract extras from nested image_url dict
        image_url_known_keys = {"url"}
        image_url_extras = _extract_extras(block["image_url"], image_url_known_keys)

        # Merge extras
        all_extras = {**extras}
        for key, value in image_url_extras.items():
            if key == "detail":  # Don't rename
                all_extras["detail"] = value
            else:
                all_extras[f"image_url_{key}"] = value

        return create_image_block(
            # Even though this is labeled as `url`, it can be base64-encoded
            base64=block["image_url"]["url"],
            mime_type=parsed["mime_type"],
            **all_extras,
        )

    # url-style image block
    if (block["type"] == "image_url") and isinstance(
        block["image_url"].get("url"), str
    ):
        known_keys = {"type", "image_url"}
        extras = _extract_extras(block, known_keys)

        image_url_known_keys = {"url"}
        image_url_extras = _extract_extras(block["image_url"], image_url_known_keys)

        all_extras = {**extras}
        for key, value in image_url_extras.items():
            if key == "detail":  # Don't rename
                all_extras["detail"] = value
            else:
                all_extras[f"image_url_{key}"] = value

        return create_image_block(
            url=block["image_url"]["url"],
            **all_extras,
        )

    # base64-style audio block
    # audio is only represented via raw data, no url or ID option
    if block["type"] == "input_audio":
        known_keys = {"type", "input_audio"}
        extras = _extract_extras(block, known_keys)

        # Also extract extras from nested audio dict
        audio_known_keys = {"data", "format"}
        audio_extras = _extract_extras(block["input_audio"], audio_known_keys)

        all_extras = {**extras}
        for key, value in audio_extras.items():
            all_extras[f"audio_{key}"] = value

        return create_audio_block(
            base64=block["input_audio"]["data"],
            mime_type=f"audio/{block['input_audio']['format']}",
            **all_extras,
        )

    # id-style file block
    if block.get("type") == "file" and "file_id" in block.get("file", {}):
        known_keys = {"type", "file"}
        extras = _extract_extras(block, known_keys)

        file_known_keys = {"file_id"}
        file_extras = _extract_extras(block["file"], file_known_keys)

        all_extras = {**extras}
        for key, value in file_extras.items():
            all_extras[f"file_{key}"] = value

        return create_file_block(
            file_id=block["file"]["file_id"],
            **all_extras,
        )

    # base64-style file block
    if (block["type"] == "file") and (
        parsed := _parse_data_uri(block["file"]["file_data"])
    ):
        known_keys = {"type", "file"}
        extras = _extract_extras(block, known_keys)

        file_known_keys = {"file_data", "filename"}
        file_extras = _extract_extras(block["file"], file_known_keys)

        all_extras = {**extras}
        for key, value in file_extras.items():
            all_extras[f"file_{key}"] = value

        mime_type = parsed["mime_type"]
        filename = block["file"].get("filename")
        return create_file_block(
            base64=block["file"]["file_data"],
            mime_type=mime_type,
            filename=filename,
            **all_extras,
        )

    # Escape hatch
    return block


def _normalize_messages_v0(messages: Sequence["BaseMessage"]) -> list["BaseMessage"]:
    """Normalize messages using the legacy v0 logic from master.

    This replicates the exact normalization logic that was in master branch
    to ensure backward compatibility with partner packages that expect v0 format.

    Only converts OpenAI file and audio formats to v0 format.
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
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    # Convert using the original master logic
                    converted_block = _convert_openai_to_v0_format(block)
                    _update_content_block(formatted_message, idx, converted_block)
        formatted_messages.append(formatted_message)

    return formatted_messages


def _convert_openai_to_v0_format(block: dict) -> dict:
    """Convert OpenAI format to v0 format using master's original logic."""
    if block["type"] == "file":
        parsed = _parse_data_uri(block["file"]["file_data"])
        if parsed is not None:
            # Create a new dict with the parsed data plus additional fields
            result = dict(parsed)
            result["type"] = "file"
            if filename := block["file"].get("filename"):
                result["filename"] = filename
            return result
        return block

    if block["type"] == "input_audio":
        audio = block.get("input_audio", {})
        data = audio.get("data")
        audio_format = audio.get("format")
        if data and audio_format:
            return {
                "type": "audio",
                "source_type": "base64",
                "data": data,
                "mime_type": f"audio/{audio_format}",
            }
        return block

    return block


def _normalize_messages(
    messages: Sequence["BaseMessage"],
    convert_all: bool = False,  # noqa: FBT001, FBT002
) -> list["BaseMessage"]:
    """Normalize message formats to LangChain v1 standard content blocks.

    Chat models already implement support for:
    - Images in OpenAI Chat Completions format
        These will be passed through unchanged unless `convert_all` is True, in which
        case they will be converted to LangChain v1 standard content blocks.
    - LangChain v1 standard content blocks

    This function extends support to:
    - `Audio <https://platform.openai.com/docs/api-reference/chat/create>`__ and
        `file <https://platform.openai.com/docs/api-reference/files>`__ data in OpenAI
        Chat Completions format
        - Images are technically supported but we expect chat models to handle them
            directly; this may change in the future
    - LangChain v0 standard content blocks for backward compatibility

    .. versionchanged:: 1.0.0
        In previous versions, this function returned messages in LangChain v0 format.
        Now, it returns messages in LangChain v1 format, which upgraded chat models now
        expect to receive when passing back in message history. For backward
        compatibility, this function will convert v0 message content to v1 format.

    .. dropdown:: v0 Content Block Schemas

        ``URLContentBlock``:

        .. codeblock::

            {
                mime_type: NotRequired[str]
                type: Literal['image', 'audio', 'file'],
                source_type: Literal['url'],
                url: str,
            }

        ``Base64ContentBlock``:

        .. codeblock::

            {
                mime_type: NotRequired[str]
                type: Literal['image', 'audio', 'file'],
                source_type: Literal['base64'],
                data: str,
            }

        ``IDContentBlock``:

        (In practice, this was never used)

        .. codeblock::

            {
                type: Literal['image', 'audio', 'file'],
                source_type: Literal['id'],
                id: str,
            }

        ``PlainTextContentBlock``:

        .. codeblock::

            {
                mime_type: NotRequired[str]
                type: Literal['file'],
                source_type: Literal['text'],
                url: str,
            }

    If a v1 message is passed in, it will be returned as-is, meaning it is safe to
    always pass in v1 messages to this function for assurance.

    """
    # For posterity, here are the OpenAI Chat Completions schemas we expect:
    #
    # Chat Completions image. Can be URL-based or base64-encoded. Supports MIME types
    # png, jpeg/jpg, webp, static gif:
    # {
    #     "type": Literal['image_url'],
    #     "image_url": {
    #         "url": Union["data:$MIME_TYPE;base64,$BASE64_ENCODED_IMAGE", "$IMAGE_URL"],  # noqa: E501
    #         "detail": Literal['low', 'high', 'auto'] = 'auto',  # Only supported by OpenAI  # noqa: E501
    #     }
    # }

    # Chat Completions audio:
    # {
    #     "type": Literal['input_audio'],
    #     "input_audio": {
    #         "format": Literal['wav', 'mp3'],
    #         "data": str = "$BASE64_ENCODED_AUDIO",
    #     },
    # }

    # Chat Completions files: either base64 or pre-uploaded file ID
    # {
    #     "type": Literal['file'],
    #     "file": Union[
    #         {
    #             "filename": Optional[str] = "$FILENAME",
    #             "file_data": str = "$BASE64_ENCODED_FILE",
    #         },
    #         {
    #             "file_id": str = "$FILE_ID",  # For pre-uploaded files to OpenAI
    #         },
    #     ],
    # }

    formatted_messages = []
    for message in messages:
        # We preserve input messages - the caller may reuse them elsewhere and expects
        # them to remain unchanged. We only create a copy if we need to translate.
        formatted_message = message

        if isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                # OpenAI Chat Completions multimodal data blocks to v1 standard
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"image_url", "input_audio", "file"}
                    # Discriminate between OpenAI/LC format since they share `'type'`
                    and _is_openai_data_block(block)
                ):
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    # Convert OpenAI audio/file block to LC v1 std content
                    # unless `all` is True, in which case we also conver images
                    if convert_all or block["type"] != "image_url":
                        converted_block = _convert_openai_format_to_data_block(block)
                        _update_content_block(formatted_message, idx, converted_block)
                    else:
                        # If `all` is False, we pass through images unchanged
                        _update_content_block(formatted_message, idx, block)

                # Convert multimodal LangChain v0 to v1 standard content blocks
                elif (
                    isinstance(block, dict)
                    and block.get("type")
                    in {
                        "image",
                        "audio",
                        "file",
                    }
                    and block.get("source_type")  # v1 doesn't have `source_type`
                    in {
                        "url",
                        "base64",
                        "id",
                        "text",
                    }
                ):
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    converted_block = _convert_legacy_v0_content_block_to_v1(block)
                    _update_content_block(formatted_message, idx, converted_block)
                    continue

                # Pass through blocks that look like they have v1 format unchanged
                elif isinstance(block, dict) and block.get("type") in KNOWN_BLOCK_TYPES:
                    # No conversion needed for v1 blocks, so no need to copy
                    pass

        # If we didn't modify the message, skip creating a new instance
        # e.g. passing through content that is just `str`
        if formatted_message is message:
            formatted_messages.append(message)
            continue

        formatted_messages.append(formatted_message)

    return formatted_messages


T = TypeVar("T", bound="BaseMessage")


def _ensure_message_copy(message: T, formatted_message: T) -> T:
    """Create a copy of the message if it hasn't been copied yet."""
    if formatted_message is message:
        formatted_message = message.model_copy()
        # Shallow-copy content list to allow modifications
        formatted_message.content = list(formatted_message.content)
    return formatted_message


def _update_content_block(
    formatted_message: "BaseMessage", idx: int, new_block: Union[ContentBlock, dict]
) -> None:
    """Update a content block at the given index, handling type issues."""
    # Type ignore needed because:
    # - `BaseMessage.content` is typed as `Union[str, list[Union[str, dict]]]`
    # - When content is str, indexing fails (index error)
    # - When content is list, the items are `Union[str, dict]` but we're assigning
    #   `Union[ContentBlock, dict]` where ContentBlock is richer than dict
    # - This is safe because we only call this when we've verified content is a list and
    #   we're doing content block conversions
    formatted_message.content[idx] = new_block  # type: ignore[index, assignment]


def _update_message_content_to_blocks(message: T, output_version: str) -> T:
    return message.model_copy(
        update={
            "content": message.content_blocks,
            "response_metadata": {
                **message.response_metadata,
                "output_version": output_version,
            },
        }
    )


def _convert_legacy_v0_content_block_to_v1(block: dict) -> ContentBlock:
    """Convert a LangChain v0 content block to v1 format.

    Preserves unknown keys as extras to avoid data loss.

    """

    def _extract_v0_extras(block_dict: dict, known_keys: set[str]) -> dict[str, Any]:
        """Extract unknown keys from v0 block to preserve as extras."""
        return {k: v for k, v in block_dict.items() if k not in known_keys}

    if block.get("type") == "image":
        source_type = block.get("source_type")
        if source_type == "url":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return create_image_block(
                url=block["url"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "base64":
            known_keys = {"type", "source_type", "data", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return create_image_block(
                base64=block["data"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "id":
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            return create_image_block(file_id=block["id"], **extras)
    elif block.get("type") == "audio":
        source_type = block.get("source_type")
        if source_type == "url":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return create_audio_block(
                url=block["url"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "base64":
            known_keys = {"type", "source_type", "data", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return create_audio_block(
                base64=block["data"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "id":
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            return create_audio_block(file_id=block["id"], **extras)
    elif block.get("type") == "file":
        source_type = block.get("source_type")
        if source_type == "url":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return create_file_block(
                url=block["url"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "base64":
            known_keys = {"type", "source_type", "data", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return create_file_block(
                base64=block["data"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "id":
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            return create_file_block(file_id=block["id"], **extras)
        if source_type == "text":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return create_plaintext_block(
                # In v0, URL points to the text file content
                text=block["url"],
                **extras,
            )

    # If we can't convert, return the block as-is
    return cast("ContentBlock", block)
