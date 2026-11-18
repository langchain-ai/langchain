"""Normalize provider finish reasons onto a shared vocabulary.

Normalized values follow the OpenTelemetry GenAI `finish_reason` values, plus
`"other"` for anything unmapped. The raw provider value is always kept
alongside; this only adds a portable view of it.
"""

from __future__ import annotations

from typing import Any, Literal

FinishReason = Literal[
    "stop", "length", "tool_call", "content_filter", "error", "other"
]
"""Provider-agnostic reason a model stopped generating."""

FinishState = Literal["complete", "suspended"]
"""Whether the turn is done or the caller must continue it (e.g. `pause_turn`)."""

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
    "refusal": "content_filter",
    "model_context_window_exceeded": "length",
    "pause_turn": "other",
}

_GEMINI: dict[str, FinishReason] = {
    "stop": "stop",
    "max_tokens": "length",
    "safety": "content_filter",
    "recitation": "content_filter",
    "blocklist": "content_filter",
    "prohibited_content": "content_filter",
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
) -> tuple[FinishReason | None, FinishState | None]:
    """Map a message's raw finish value onto `FinishReason`.

    Args:
        response_metadata: The message's response metadata. The provider is
            read from `model_provider`; the raw value from `finish_reason` or
            `stop_reason` (or `status` for the OpenAI Responses API).
        has_tool_calls: Whether the message carries tool calls. A stop-like
            value is reported as `"tool_call"` when `True`.

    Returns:
        `(normalized, state)`. Both are `None` when no raw value is present;
        unknown values map to `"other"`, never `"stop"`.
    """
    metadata = response_metadata or {}
    provider = metadata.get("model_provider")
    has_raw_reason = (
        metadata.get("finish_reason") is not None
        or metadata.get("stop_reason") is not None
    )

    normalized: FinishReason | None
    state: FinishState = "complete"
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
    if normalized == "stop" and has_tool_calls:
        normalized = "tool_call"
    return normalized, state
