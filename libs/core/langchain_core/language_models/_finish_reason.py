"""Normalize provider finish reasons onto a shared vocabulary.

Unmapped values become `"other"`. The raw provider value is always kept
alongside; this only adds a portable view of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_protocol.protocol import FinishReason, FinishState

    from langchain_core.messages import BaseMessage


_OPENAI_CHAT: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "content_filter",
}

_ANTHROPIC: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_call",
    "refusal": "refusal",
    "model_context_window_exceeded": "context_exceeded",
    "pause_turn": "other",
}

_GEMINI: dict[str, FinishReason] = {
    "stop": "stop",
    "max_tokens": "length",
    "safety": "refusal",
    "recitation": "content_filter",
    "blocklist": "refusal",
    "prohibited_content": "refusal",
    "spii": "content_filter",
    "language": "content_filter",
    "image_safety": "content_filter",
    "image_prohibited_content": "content_filter",
    "image_recitation": "content_filter",
}

_RESPONSES_INCOMPLETE: dict[str, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}

_GEMINI_PROVIDERS = frozenset({"google_genai", "google_vertexai"})

_BEDROCK: dict[str, FinishReason] = {
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
}

# Unknown providers: try every known vocabulary.
_GENERIC: dict[str, FinishReason] = {
    **_GEMINI,
    **_ANTHROPIC,
    **_BEDROCK,
    **_OPENAI_CHAT,
}


def get_raw_finish_reason(response_metadata: dict[str, Any]) -> Any:
    """Return the provider's raw finish value, or `None` if absent.

    Args:
        response_metadata: The message's response metadata.

    Returns:
        The value of `finish_reason`, falling back to `stop_reason`, then
        `status` for the OpenAI Responses API.
    """
    raw = response_metadata.get("finish_reason")
    if raw is None:
        raw = response_metadata.get("stop_reason")
    if raw is None and response_metadata.get("model_provider") == "openai":
        # OpenAI Responses reports `status` instead of `finish_reason`.
        raw = response_metadata.get("status")
    return raw


def _normalize_responses(
    response_metadata: dict[str, Any],
) -> FinishReason | None:
    status = response_metadata.get("status")
    if status == "completed":
        return "stop"
    if status == "incomplete":
        details = response_metadata.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        return _RESPONSES_INCOMPLETE.get(str(reason), "other")
    if status == "failed":
        return "error"
    if status is None:
        return None
    return "other"


def normalize_finish_reason(
    response_metadata: dict[str, Any] | None,
    *,
    has_tool_calls: bool = False,
    has_refusal: bool = False,
) -> tuple[FinishReason | None, FinishState | None]:
    """Map a message's raw finish value onto `FinishReason`.

    Args:
        response_metadata: The message's response metadata. The provider is
            read from `model_provider`; the raw value from `finish_reason` or
            `stop_reason` (or `status` for the OpenAI Responses API).
        has_tool_calls: Whether the message carries tool calls. A stop-like
            value is reported as `"tool_call"` when `True`.
        has_refusal: Whether the message carries a `refusal` content block.
            Reported as `"refusal"` when `True`, since OpenAI refuses with a
            normal `stop` / `completed` value.

    Returns:
        `(normalized, state)`. Both are `None` when no raw value is present;
        unknown values map to `"other"`, never `"stop"`. `state` is set only
        for Anthropic (and `pause_turn`).
    """
    metadata = response_metadata or {}
    provider = metadata.get("model_provider")
    has_raw_reason = (
        metadata.get("finish_reason") is not None
        or metadata.get("stop_reason") is not None
    )

    normalized: FinishReason | None
    # Only Anthropic can suspend a turn, so only it reports a finish state.
    state: FinishState | None = "complete" if provider == "anthropic" else None
    if not has_raw_reason:
        normalized = _normalize_responses(metadata) if provider == "openai" else None
    else:
        key = str(get_raw_finish_reason(metadata)).lower().replace("-", "_")
        if key == "pause_turn":
            state = "suspended"
        if provider == "anthropic":
            normalized = _ANTHROPIC.get(key, "other")
        elif provider in _GEMINI_PROVIDERS:
            normalized = _GEMINI.get(key, "other")
        elif provider == "openai":
            normalized = _OPENAI_CHAT.get(key, "other")
        else:
            normalized = _GENERIC.get(key, "other")

    if normalized is None:
        return None, None
    if has_refusal:
        return "refusal", state
    if normalized == "stop" and has_tool_calls:
        normalized = "tool_call"
    return normalized, state


def add_finish_reason_fields(
    response_metadata: dict[str, Any],
    *,
    has_tool_calls: bool,
    has_refusal: bool = False,
) -> None:
    """Add `finish_reason`, `normalized_finish_reason` and `finish_state` in place.

    `finish_state` is added only for Anthropic.

    Existing values are kept, so fields set by the streaming path win.

    Args:
        response_metadata: The message's response metadata.
        has_tool_calls: Whether the message carries tool calls.
        has_refusal: Whether the message carries a `refusal` content block.
    """
    raw = get_raw_finish_reason(response_metadata)
    normalized, state = normalize_finish_reason(
        response_metadata, has_tool_calls=has_tool_calls, has_refusal=has_refusal
    )
    if raw is not None:
        response_metadata.setdefault("finish_reason", raw)
    if normalized is not None:
        response_metadata.setdefault("normalized_finish_reason", normalized)
    if state is not None:
        response_metadata.setdefault("finish_state", state)


def has_refusal_block(message: BaseMessage) -> bool:
    """Return whether a message's content blocks include a `refusal` block."""
    return any(block.get("type") == "refusal" for block in message.content_blocks)
