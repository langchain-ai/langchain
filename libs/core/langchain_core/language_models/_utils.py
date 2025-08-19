import re
from collections.abc import Sequence
from typing import Literal, Optional, TypedDict, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.messages.content import (
    KNOWN_BLOCK_TYPES,
    ContentBlock,
    create_audio_block,
    create_file_block,
    create_image_block,
    create_non_standard_block,
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
        if (audio := block.get("audio")) and isinstance(audio, dict):
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


def _convert_openai_format_to_data_block(block: dict) -> ContentBlock:
    """Convert OpenAI image/audio/file content block to respective v1 multimodal block.

    We expect that the incoming block is verified to be in OpenAI Chat Completions
    format.

    If parsing fails, passes block through unchanged.

    Mappings (Chat Completions to LangChain v1):
    - Image -> `ImageContentBlock`
    - Audio -> `AudioContentBlock`
    - File -> `FileContentBlock`

    """
    # TODO: if an incoming block has extra keys, should we ignore them or stuff
    # into extras?

    # base64-style image block
    if (block["type"] == "image_url") and (
        parsed := _parse_data_uri(block["image_url"]["url"])
    ):
        return create_image_block(
            # Even though this is labeled as `url`, it can be base64-encoded
            base64=block["image_url"]["url"],
            mime_type=parsed["mime_type"],
            detail=block["image_url"].get("detail"),  # Optional, specific to OpenAI
        )

    # url-style image block
    if (block["type"] == "image_url") and isinstance(
        block["image_url"].get("url"), str
    ):
        return create_image_block(
            url=block["image_url"]["url"],
            detail=block["image_url"].get("detail"),  # Optional, specific to OpenAI
        )

    # base64-style audio block
    # audio is only represented via raw data, no url or ID option
    if block["type"] == "input_audio":
        return create_audio_block(
            base64=block["audio"]["data"],
            mime_type=f"audio/{block['audio']['format']}",
        )

    # id-style file block
    if block.get("type") == "file" and "file_id" in block.get("file", {}):
        return create_file_block(
            file_id=block["file"]["file_id"],
        )

    # base64-style file block
    if (block["type"] == "file") and (
        parsed := _parse_data_uri(block["file"]["file_data"])
    ):
        mime_type = parsed["mime_type"]
        filename = block["file"].get("filename")
        return create_file_block(
            base64=block["file"]["file_data"],
            mime_type=mime_type,
            filename=filename,
        )

    # Escape hatch for non-standard content blocks
    return create_non_standard_block(
        value=block,
    )


def _normalize_messages(messages: Sequence["BaseMessage"]) -> list["BaseMessage"]:
    """Normalize message formats to LangChain v1 standard content blocks.

    Chat models already implement support for:
    - Images in OpenAI Chat Completions format
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

    (Untested): if a v1 message is passed in, it will be returned as-is, meaning it is
    safe to always pass in v1 messages to this function for assurance.

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
    #     "audio": {
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

        if isinstance(message.content, str):
            if formatted_message is message:
                formatted_message = message.model_copy()
                # Shallow-copy the content string so we can modify it
                formatted_message.content = str(formatted_message.content)
            formatted_message.content = [
                {
                    "type": "text",
                    "text": message.content,
                },
                # TODO: decide
                # could use factory instead, to auto-gen an ID for the block?
                #
                # create_text_block(
                #     text=message.content,
                # ),
            ]

        elif isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                if isinstance(block, str):
                    if formatted_message is message:
                        formatted_message = message.model_copy()
                        formatted_message.content = list(formatted_message.content)

                    # TODO: same as above, could use factory instead
                    formatted_message.content[idx] = {"type": "text", "text": block}  # type: ignore[index]  # mypy confused by .model_copy

                # OpenAI Chat Completions multimodal data blocks to v1 standard
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"image_url", "input_audio", "file"}
                    # Discriminate between OpenAI/LC format since they share `'type'`
                    and _is_openai_data_block(block)
                ):
                    if formatted_message is message:
                        formatted_message = message.model_copy()
                        formatted_message.content = list(formatted_message.content)

                    # Convert OpenAI image/audio/file block to LC v1 std content
                    formatted_message.content[idx] = (  # type: ignore[call-overload,index]
                        _convert_openai_format_to_data_block(block)
                        # Note: this may assign a NonStandardCB if parsing fails
                    )

                # Convert LangChain v0 to v1 standard content blocks
                elif (
                    isinstance(block, dict)
                    and block.get("type")
                    in {
                        # Superset of possible v0 content block `type`s
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
                    if formatted_message is message:
                        formatted_message = message.model_copy()
                        formatted_message.content = list(formatted_message.content)

                    if block["source_type"] == "url" and block["type"] == "image":
                        formatted_message.content[idx] = create_image_block(  # type: ignore[call-overload,index]
                            url=block["url"],
                            mime_type=block.get("mime_type"),
                        )
                    elif block["source_type"] == "base64" and block["type"] == "image":
                        formatted_message.content[idx] = create_image_block(  # type: ignore[call-overload,index]
                            base64=block["data"],
                            mime_type=block.get("mime_type"),
                        )
                    elif block["source_type"] == "id" and block["type"] == "image":
                        formatted_message.content[idx] = create_image_block(  # type: ignore[call-overload,index]
                            id=block["id"],
                        )
                    elif block["source_type"] == "url" and block["type"] == "audio":
                        formatted_message.content[idx] = create_audio_block(  # type: ignore[call-overload,index]
                            url=block["url"],
                            mime_type=block.get("mime_type"),
                        )
                    elif block["source_type"] == "base64" and block["type"] == "audio":
                        formatted_message.content[idx] = create_audio_block(  # type: ignore[call-overload,index]
                            base64=block["data"],
                            mime_type=block.get("mime_type"),
                        )
                    elif block["source_type"] == "id" and block["type"] == "audio":
                        formatted_message.content[idx] = create_audio_block(  # type: ignore[call-overload,index]
                            id=block["id"],
                        )
                    elif block["source_type"] == "url" and block["type"] == "file":
                        formatted_message.content[idx] = create_file_block(  # type: ignore[call-overload,index]
                            url=block["url"],
                            mime_type=block.get("mime_type"),
                        )
                    elif block["source_type"] == "base64" and block["type"] == "file":
                        formatted_message.content[idx] = create_file_block(  # type: ignore[call-overload,index]
                            base64=block["data"],
                            mime_type=block.get("mime_type"),
                        )
                    elif block["source_type"] == "id" and block["type"] == "file":
                        formatted_message.content[idx] = create_file_block(  # type: ignore[call-overload,index]
                            id=block["id"],
                        )
                    elif block["source_type"] == "text" and block["type"] == "file":
                        formatted_message.content[idx] = create_plaintext_block(  # type: ignore[call-overload,index]
                            # In v0, URL points to the text file content
                            text=block["url"],
                        )
                    else:
                        # Unsupported or malformed v0 content block
                        formatted_message.content[idx] = {  # type: ignore[index]
                            "type": "non_standard",
                            "value": block,
                        }

                # Pass through blocks that look like they have v1 format unchanged
                elif isinstance(block, dict) and block.get("type") in KNOWN_BLOCK_TYPES:
                    formatted_message.content[idx] = block  # type: ignore[index]

        # If we didn't modify the message, skip creating a new instance
        if formatted_message is message:
            formatted_messages.append(message)
            continue

        # At this point, `content` will be a list of v1 standard content blocks.
        formatted_messages.append(formatted_message)

    return formatted_messages


T = TypeVar("T", bound=BaseMessage)


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
