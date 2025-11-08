import re
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Literal,
    TypedDict,
    TypeVar,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
from langchain_core.messages.content import (
    ContentBlock,
)


def is_openai_data_block(
    block: dict, filter_: Literal["image", "audio", "file"] | None = None
) -> bool:
    """Check whether a block contains multimodal data in OpenAI Chat Completions format.

    Supports both data and ID-style blocks (e.g. `'file_data'` and `'file_id'`)

    If additional keys are present, they are ignored / will not affect outcome as long
    as the required keys are present and valid.

    Args:
        block: The content block to check.
        filter_: If provided, only return True for blocks matching this specific type.
            - "image": Only match image_url blocks
            - "audio": Only match input_audio blocks
            - "file": Only match file blocks
            If `None`, match any valid OpenAI data block type. Note that this means that
            if the block has a valid OpenAI data type but the filter_ is set to a
            different type, this function will return False.

    Returns:
        `True` if the block is a valid OpenAI data block and matches the filter_
        (if provided).

    """
    if block.get("type") == "image_url":
        if filter_ is not None and filter_ != "image":
            return False
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
        if filter_ is not None and filter_ != "audio":
            return False
        if (audio := block.get("input_audio")) and isinstance(audio, dict):
            audio_data = audio.get("data")
            audio_format = audio.get("format")
            # Both required per OpenAI spec
            if isinstance(audio_data, str) and isinstance(audio_format, str):
                return True

    elif block.get("type") == "file":
        if filter_ is not None and filter_ != "file":
            return False
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


def _parse_data_uri(uri: str) -> ParsedDataUri | None:
    """Parse a data URI into its components.

    If parsing fails, return `None`. If either MIME type or data is missing, return
    `None`.

    Example:
        ```python
        data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        parsed = _parse_data_uri(data_uri)

        assert parsed == {
            "source_type": "base64",
            "mime_type": "image/jpeg",
            "data": "/9j/4AAQSkZJRg...",
        }
        ```
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


def _normalize_messages(
    messages: Sequence["BaseMessage"],
) -> list["BaseMessage"]:
    """Normalize message formats to LangChain v1 standard content blocks.

    Chat models already implement support for:
    - Images in OpenAI Chat Completions format
        These will be passed through unchanged
    - LangChain v1 standard content blocks

    This function extends support to:
    - `[Audio](https://platform.openai.com/docs/api-reference/chat/create) and
        `[file](https://platform.openai.com/docs/api-reference/files) data in OpenAI
        Chat Completions format
        - Images are technically supported but we expect chat models to handle them
            directly; this may change in the future
    - LangChain v0 standard content blocks for backward compatibility

    !!! warning "Behavior changed in `langchain-core` 1.0.0"
        In previous versions, this function returned messages in LangChain v0 format.
        Now, it returns messages in LangChain v1 format, which upgraded chat models now
        expect to receive when passing back in message history. For backward
        compatibility, this function will convert v0 message content to v1 format.

    ??? note "v0 Content Block Schemas"

        `URLContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['image', 'audio', 'file'],
            source_type: Literal['url'],
            url: str,
        }
        ```

        `Base64ContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['image', 'audio', 'file'],
            source_type: Literal['base64'],
            data: str,
        }
        ```

        `IDContentBlock`:

        (In practice, this was never used)

        ```python
        {
            type: Literal["image", "audio", "file"],
            source_type: Literal["id"],
            id: str,
        }
        ```

        `PlainTextContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['file'],
            source_type: Literal['text'],
            url: str,
        }
        ```

    If a v1 message is passed in, it will be returned as-is, meaning it is safe to
    always pass in v1 messages to this function for assurance.

    For posterity, here are the OpenAI Chat Completions schemas we expect:

    Chat Completions image. Can be URL-based or base64-encoded. Supports MIME types
    png, jpeg/jpg, webp, static gif:
    {
        "type": Literal['image_url'],
        "image_url": {
            "url": Union["data:$MIME_TYPE;base64,$BASE64_ENCODED_IMAGE", "$IMAGE_URL"],
            "detail": Literal['low', 'high', 'auto'] = 'auto',  # Supported by OpenAI
        }
    }

    Chat Completions audio:
    {
        "type": Literal['input_audio'],
        "input_audio": {
            "format": Literal['wav', 'mp3'],
            "data": str = "$BASE64_ENCODED_AUDIO",
        },
    }

    Chat Completions files: either base64 or pre-uploaded file ID
    {
        "type": Literal['file'],
        "file": Union[
            {
                "filename": str | None = "$FILENAME",
                "file_data": str = "$BASE64_ENCODED_FILE",
            },
            {
                "file_id": str = "$FILE_ID",  # For pre-uploaded files to OpenAI
            },
        ],
    }

    """
    from langchain_core.messages.block_translators.langchain_v0 import (  # noqa: PLC0415
        _convert_legacy_v0_content_block_to_v1,
    )
    from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
        _convert_openai_format_to_data_block,
    )

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
                    and block.get("type") in {"input_audio", "file"}
                    # Discriminate between OpenAI/LC format since they share `'type'`
                    and is_openai_data_block(block)
                ):
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    converted_block = _convert_openai_format_to_data_block(block)
                    _update_content_block(formatted_message, idx, converted_block)

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

                # else, pass through blocks that look like they have v1 format unchanged

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
    formatted_message: "BaseMessage", idx: int, new_block: ContentBlock | dict
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
