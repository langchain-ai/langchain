"""Read the finish fields that provider integrations set on `response_metadata`.

Providers own the mapping from their raw values: they set `finish_reason` (raw),
`normalized_finish_reason` and, when relevant, `finish_state`. Core only reads
them and applies two provider-agnostic corrections.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk, BaseMessageChunk

_FINISH_KEYS = ("finish_reason", "normalized_finish_reason", "finish_state")


def _finish_fields(
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


class _StreamFinishCorrector:
    """Applies the corrections to the final chunk of a `stream()` call.

    The finish reason arrives on the last chunk, after any tool calls or refusal.
    """

    def __init__(self) -> None:
        self._saw_tool_call = False
        self._saw_refusal = False

    def update(self, message: BaseMessageChunk) -> None:
        if isinstance(message, AIMessageChunk) and message.tool_call_chunks:
            self._saw_tool_call = True
        if message.additional_kwargs.get("refusal") or (
            isinstance(message.content, list)
            and any(
                isinstance(block, dict) and block.get("type") == "refusal"
                for block in message.content
            )
        ):
            self._saw_refusal = True
        metadata = message.response_metadata
        if metadata.get("normalized_finish_reason"):
            metadata.update(
                _finish_fields(
                    metadata,
                    has_tool_calls=self._saw_tool_call,
                    has_refusal=self._saw_refusal,
                )
            )
