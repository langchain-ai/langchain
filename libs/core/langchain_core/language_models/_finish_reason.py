"""Read the finish fields that provider integrations set on `response_metadata`.

Providers own the mapping from their raw values: they set `finish_reason` (raw),
`normalized_finish_reason` and, when relevant, `finish_state`. Core only reads
them and applies two provider-agnostic corrections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

_FINISH_KEYS = ("finish_reason", "normalized_finish_reason", "finish_state")


def finish_fields(
    response_metadata: dict[str, Any] | None,
    *,
    has_tool_calls: bool = False,
    has_refusal: bool = False,
) -> dict[str, Any]:
    """Return the finish fields a provider set, with generic corrections.

    Args:
        response_metadata: The message's response metadata.
        has_tool_calls: Whether the message carries tool calls. A normalized
            `"stop"` becomes `"tool_call"` when `True`.
        has_refusal: Whether the message carries a `refusal` content block.
            The normalized value becomes `"refusal"` when `True`.

    Returns:
        The finish fields present in `response_metadata`, empty when the
        provider set none.
    """
    metadata = response_metadata or {}
    fields = {key: metadata[key] for key in _FINISH_KEYS if metadata.get(key)}
    normalized = fields.get("normalized_finish_reason")
    if normalized is None:
        return fields
    # Refusals and tool calls can arrive in earlier stream chunks than the
    # finish reason, so correct them on the assembled message.
    if has_refusal:
        fields["normalized_finish_reason"] = "refusal"
    elif normalized == "stop" and has_tool_calls:
        fields["normalized_finish_reason"] = "tool_call"
    return fields


def has_refusal_block(message: BaseMessage) -> bool:
    """Return whether a message's content blocks include a `refusal` block."""
    return any(block.get("type") == "refusal" for block in message.content_blocks)
