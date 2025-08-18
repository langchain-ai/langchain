"""Derivations of standard content blocks from Anthropic content."""

from collections.abc import Iterable
from typing import Any, cast

from langchain_core.messages import content_blocks as types


def _populate_extras(
    standard_block: types.ContentBlock, block: dict[str, Any], known_fields: set[str]
) -> types.ContentBlock:
    """Mutate a block, populating extras."""
    if standard_block.get("type") == "non_standard":
        return standard_block

    for key, value in block.items():
        if key not in known_fields:
            if "extras" not in block:
                # Below type-ignores are because mypy thinks a non-standard block can
                # get here, although we exclude them above.
                standard_block["extras"] = {}  # type: ignore[typeddict-unknown-key]
            standard_block["extras"][key] = value  # type: ignore[typeddict-item]

    return standard_block


def _convert_to_v1_from_anthropic_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Attempt to unpack non-standard blocks."""

    def _iter_blocks() -> Iterable[types.ContentBlock]:
        blocks: list[dict[str, Any]] = [
            cast("dict[str, Any]", block)
            if block.get("type") != "non_standard"
            else block["value"]  # type: ignore[typeddict-item]  # this is only non-standard blocks
            for block in content
        ]
        for block in blocks:
            block_type = block.get("type")

            if (
                block_type == "document"
                and "source" in block
                and "type" in block["source"]
            ):
                if block["source"]["type"] == "base64":
                    file_block: types.FileContentBlock = {
                        "type": "file",
                        "base64": block["source"]["data"],
                        "mime_type": block["source"]["media_type"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif block["source"]["type"] == "url":
                    file_block = {
                        "type": "file",
                        "url": block["source"]["url"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif block["source"]["type"] == "file":
                    file_block = {
                        "type": "file",
                        "id": block["source"]["file_id"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif block["source"]["type"] == "text":
                    plain_text_block: types.PlainTextContentBlock = {
                        "type": "text-plain",
                        "text": block["source"]["data"],
                        "mime_type": block.get("media_type", "text/plain"),
                    }
                    _populate_extras(plain_text_block, block, {"type", "source"})
                    yield plain_text_block

                else:
                    yield {"type": "non_standard", "value": block}

            elif (
                block_type == "image"
                and "source" in block
                and "type" in block["source"]
            ):
                if block["source"]["type"] == "base64":
                    image_block: types.ImageContentBlock = {
                        "type": "image",
                        "base64": block["source"]["data"],
                        "mime_type": block["source"]["media_type"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                elif block["source"]["type"] == "url":
                    image_block = {
                        "type": "image",
                        "url": block["source"]["url"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                elif block["source"]["type"] == "file":
                    image_block = {
                        "type": "image",
                        "id": block["source"]["file_id"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                else:
                    yield {"type": "non_standard", "value": block}

            elif block_type in types.KNOWN_BLOCK_TYPES:
                yield cast("types.ContentBlock", block)

            else:
                yield {"type": "non_standard", "value": block}

    return list(_iter_blocks())
