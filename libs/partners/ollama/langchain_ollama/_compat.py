"""Go from v1 content blocks to Ollama SDK format."""

from typing import Any

from langchain_core.messages import content as types


def _convert_from_v1_to_ollama(
    content: list[types.ContentBlock],
    model_provider: str | None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Convert v1 content blocks to Ollama format.

    Args:
        content: List of v1 `ContentBlock` objects.
        model_provider: The model provider name that generated the v1 content.

    Returns:
        TODO
    """
    new_content: list = []
    for block in content:
        if not isinstance(block, dict) or "type" not in block:
            continue

        block_dict = dict(block)  # (For typing)

        # TextContentBlock
        if block_dict["type"] == "text":
            # Note: this drops all other fields/extras
            new_content.append({"type": "text", "text": block_dict["text"]})

        # ReasoningContentBlock
        # Ollama doesn't take reasoning back in
        # In the future, could consider coercing into text as an option?
        # e.g.:
        # if block_dict["type"] == "reasoning":
        #     # Attempt to preserve content in text form
        #     new_content.append({"text": str(block_dict["reasoning"])})

        # ImageContentBlock
        if block_dict["type"] == "image":
            # Already handled in _get_image_from_data_content_block
            new_content.append(block_dict)

        # TODO: AudioContentBlock once models support

        # TODO: FileContentBlock once models support

        # ToolCall -> ???
        # if block_dict["type"] == "tool_call":
        #     function_call = {}
        #     new_content.append(function_call)

        # ToolCallChunk -> ???
        # elif block_dict["type"] == "tool_call_chunk":
        #     function_call = {}
        #     new_content.append(function_call)

        # NonStandardContentBlock
        if block_dict["type"] == "non_standard":
            # Attempt to preserve content in text form
            new_content.append(
                {"type": "text", "text": str(block_dict.get("value", ""))}
            )

    return new_content
