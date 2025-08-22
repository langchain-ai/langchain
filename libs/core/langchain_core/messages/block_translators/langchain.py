"""Derivations of standard content blocks from LangChain content."""

from typing import Any, Union, cast

from langchain_core.language_models._utils import _parse_data_uri
from langchain_core.messages import content as types


def _convert_v0_multimodal_input_to_v1(
    blocks: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Convert v0 multimodal blocks to v1 format.

    Processes non_standard blocks that might be v0 format and converts them
    to proper v1 ContentBlocks.

    Args:
        blocks: List of content blocks to process.

    Returns:
        Updated list with v0 blocks converted to v1 format.
    """
    converted_blocks = []
    for block in blocks:
        if (
            isinstance(block, dict)
            and block.get("type") == "non_standard"
            and "value" in block
            and isinstance(block["value"], dict)  # type: ignore[typeddict-item]
        ):
            # We know this is a NonStandardContentBlock, so we can safely access value
            value = cast("Any", block)["value"]
            # Check if this looks like v0 format
            if (
                value.get("type") in {"image", "audio", "file"}
                and "source_type" in value
            ):
                converted_block = _convert_legacy_v0_content_block_to_v1(value)
                converted_blocks.append(cast("types.ContentBlock", converted_block))
            else:
                converted_blocks.append(block)
        else:
            converted_blocks.append(block)

    return converted_blocks


def _convert_legacy_v0_content_block_to_v1(
    block: dict,
) -> Union[types.ContentBlock, dict]:
    """Convert a LangChain v0 content block to v1 format.

    Preserves unknown keys as extras to avoid data loss.

    Returns the original block unchanged if it's not in v0 format.

    """

    def _extract_v0_extras(block_dict: dict, known_keys: set[str]) -> dict[str, Any]:
        """Extract unknown keys from v0 block to preserve as extras."""
        return {k: v for k, v in block_dict.items() if k not in known_keys}

    # Check if this is actually a v0 format block
    block_type = block.get("type")
    if block_type not in {"image", "audio", "file"} or "source_type" not in block:
        # Not a v0 format block, return unchanged
        return block

    if block.get("type") == "image":
        source_type = block.get("source_type")
        if source_type == "url":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_image_block(
                    url=block["url"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

            # Don't construct with an ID if not present in original block
            v1_block = types.ImageContentBlock(type="image", url=block["url"])
            if block.get("mime_type"):
                v1_block["mime_type"] = block["mime_type"]

            for key, value in extras.items():
                if value is not None:
                    v1_block["extras"] = {}
                    v1_block["extras"][key] = value
            return v1_block
        if source_type == "base64":
            known_keys = {"type", "source_type", "data", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_image_block(
                    base64=block["data"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

            v1_block = types.ImageContentBlock(type="image", base64=block["data"])
            if block.get("mime_type"):
                v1_block["mime_type"] = block["mime_type"]

            for key, value in extras.items():
                if value is not None:
                    v1_block["extras"] = {}
                    v1_block["extras"][key] = value
            return v1_block
        if source_type == "id":
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            # For id `source_type`, `id` is the file reference, not block ID
            v1_block = types.ImageContentBlock(type="image", file_id=block["id"])

            for key, value in extras.items():
                if value is not None:
                    v1_block["extras"] = {}
                    v1_block["extras"][key] = value

            return v1_block
    elif block.get("type") == "audio":
        source_type = block.get("source_type")
        if source_type == "url":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_audio_block(
                url=block["url"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "base64":
            known_keys = {"type", "source_type", "data", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_audio_block(
                base64=block["data"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "id":
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_audio_block(file_id=block["id"], **extras)
    elif block.get("type") == "file":
        source_type = block.get("source_type")
        if source_type == "url":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_file_block(
                url=block["url"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "base64":
            known_keys = {"type", "source_type", "data", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_file_block(
                base64=block["data"], mime_type=block.get("mime_type"), **extras
            )
        if source_type == "id":
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_file_block(file_id=block["id"], **extras)
        if source_type == "text":
            known_keys = {"type", "source_type", "url", "mime_type"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_plaintext_block(
                # In v0, URL points to the text file content
                text=block["url"],
                **extras,
            )

    # If we can't convert, return the block unchanged
    return block


def _convert_openai_format_to_data_block(
    block: dict,
) -> Union[types.ContentBlock, dict[Any, Any]]:
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

        return types.create_image_block(
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

        return types.create_image_block(
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

        return types.create_audio_block(
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

        return types.create_file_block(
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
        return types.create_file_block(
            base64=block["file"]["file_data"],
            mime_type=mime_type,
            filename=filename,
            **all_extras,
        )

    # Escape hatch
    return block
