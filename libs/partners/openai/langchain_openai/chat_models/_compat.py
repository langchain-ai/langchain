from typing import Union

from langchain_core.messages import AIMessage

_FUNCTION_CALL_IDS_MAP_KEY = "__openai_function_call_ids__"


def _convert_to_v03_ai_message(message: AIMessage) -> AIMessage:
    """Mutate an AIMessage to the old-style v0.3 format."""
    if isinstance(message.content, list):
        new_content: list[Union[dict, str]] = []
        for block in message.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    # Store a reasoning item in additional_kwargs (overwriting as in
                    # v0.3)
                    message.additional_kwargs["reasoning"] = block
                elif block.get("type") in (
                    "web_search_call",
                    "file_search_call",
                    "computer_call",
                    "code_interpreter_call",
                    "mcp_call",
                    "mcp_list_tools",
                    "mcp_approval_request",
                    "image_generation_call",
                ):
                    # Store built-in tool calls in additional_kwargs
                    if "tool_outputs" not in message.additional_kwargs:
                        message.additional_kwargs["tool_outputs"] = []
                    message.additional_kwargs["tool_outputs"].append(block)
                elif block.get("type") == "function_call" and (
                    call_id := block.get("call_id")
                    and (function_call_id := block.get("id"))
                ):
                    # Store function call item IDs in additional_kwargs, otherwise
                    # discard function call items.
                    if _FUNCTION_CALL_IDS_MAP_KEY not in message.additional_kwargs:
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {}
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][
                            call_id
                        ] = function_call_id
                elif (block.get("type") == "refusal") and (
                    refusal := block.get("refusal")
                ):
                    # Store a refusal item in additional_kwargs (overwriting as in
                    # v0.3)
                    message.additional_kwargs["refusal"] = refusal
                elif block.get("type") == "text":
                    # Store a message item ID on AIMessage.id
                    if "id" in block:
                        message.id = block["id"]
                    new_content.append({k: v for k, v in block.items() if k != "id"})
                elif (
                    set(block.keys()).issubset({"id", "index"})
                    and isinstance(block["id"], str)
                    and block["id"].startswith("msg_")
                ):
                    # Drop message IDs in streaming case
                    new_content.append({"index": block["index"]})
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        message.content = new_content
    else:
        pass

    return message


def _convert_from_v03_ai_message(message: AIMessage) -> AIMessage:
    """Convert an old-style v0.3 AIMessage into the new content-block format."""
    # Only update ChatOpenAI v0.3 AIMessages
    if not (
        isinstance(message.content, list)
        and all(isinstance(b, dict) for b in message.content)
    ) or not any(
        item in message.additional_kwargs
        for item in ["reasoning", "tool_outputs", "refusal"]
    ):
        return message

    content_order = [
        "reasoning",
        "code_interpreter_call",
        "mcp_call",
        "image_generation_call",
        "text",
        "refusal",
        "function_call",
        "computer_call",
        "mcp_list_tools",
        "mcp_approval_request",
        # N. B. "web_search_call" and "file_search_call" were not passed back in
        # in v0.3
    ]

    # Build a bucket for every known block type
    buckets: dict[str, list] = {key: [] for key in content_order}
    unknown_blocks = []

    # Reasoning
    if reasoning := message.additional_kwargs.get("reasoning"):
        buckets["reasoning"].append(reasoning)

    # Refusal
    if refusal := message.additional_kwargs.get("refusal"):
        buckets["refusal"].append({"type": "refusal", "refusal": refusal})

    # Text
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            block_copy = block.copy()
            block_copy["id"] = message.id
            buckets["text"].append(block_copy)
        else:
            unknown_blocks.append(block)

    # Tool outputs
    tool_outputs = message.additional_kwargs.get("tool_outputs", [])
    for block in tool_outputs:
        if isinstance(block, dict) and (key := block.get("type")) and key in buckets:
            buckets[key].append(block)
        else:
            unknown_blocks.append(block)

    # Re-assemble the content list in the canonical order
    new_content = []
    for key in content_order:
        new_content.extend(buckets[key])
    new_content.extend(unknown_blocks)

    new_additional_kwargs = dict(message.additional_kwargs)
    new_additional_kwargs.pop("reasoning", None)
    new_additional_kwargs.pop("refusal", None)
    new_additional_kwargs.pop("tool_outputs", None)

    if "id" in message.response_metadata:
        new_id = message.response_metadata["id"]
    else:
        new_id = message.id

    return message.model_copy(
        update={
            "content": new_content,
            "additional_kwargs": new_additional_kwargs,
            "id": new_id,
        },
        deep=False,
    )
