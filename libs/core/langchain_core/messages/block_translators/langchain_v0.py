"""Derivations of standard content blocks from LangChain v0 multimodal content."""

from typing import Any, Union, cast

from langchain_core.messages import content as types


def _convert_v0_multimodal_input_to_v1(
    blocks: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Convert v0 multimodal blocks to v1 format.

    During the `.content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a ``'non_standard'`` block with the original block stored in the ``value``
    field. This function attempts to unpack those blocks and convert any v0 format
    blocks to v1 format.

    If conversion fails, the block is left as a ``'non_standard'`` block.

    Args:
        blocks: List of content blocks to process.

    Returns:
        v1 content blocks.
    """
    converted_blocks = []
    unpacked_blocks: list[dict[str, Any]] = [
        cast("dict[str, Any]", block)
        if block.get("type") != "non_standard"
        else block["value"]  # type: ignore[typeddict-item]  # this is only non-standard blocks
        for block in blocks
    ]
    for block in unpacked_blocks:
        if block.get("type") in {"image", "audio", "file"} and "source_type" in block:
            converted_block = _convert_legacy_v0_content_block_to_v1(block)
            converted_blocks.append(cast("types.ContentBlock", converted_block))
        elif block.get("type") in types.KNOWN_BLOCK_TYPES:
            # Guard in case this function is used outside of the .content_blocks flow
            converted_blocks.append(cast("types.ContentBlock", block))
        else:
            converted_blocks.append({"type": "non_standard", "value": block})

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
