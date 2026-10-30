"""Recognize explicit provider refusal signals without changing their payloads."""

from typing import Any

_GEMINI_BLOCK_REASONS = frozenset(
    {"SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "JAILBREAK"}
)
_GEMINI_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "MODEL_ARMOR",
    }
)


def _is_refusal_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") == "non_standard":
        block = block.get("value")
    return isinstance(block, dict) and block.get("type") == "refusal"


def _is_model_refusal(
    content: str | list[str | dict[Any, Any]],
    additional_kwargs: dict[Any, Any],
    response_metadata: dict[Any, Any],
) -> bool:
    """Return whether a response contains an explicit refusal or policy block."""
    if response_metadata.get("refusal") is True or additional_kwargs.get("refusal"):
        return True
    if isinstance(content, list) and any(_is_refusal_block(block) for block in content):
        return True
    if response_metadata.get("stop_reason") == "refusal":
        return True
    finish_reason = response_metadata.get("finish_reason")
    if isinstance(finish_reason, str) and (
        finish_reason == "content_filter" or finish_reason in _GEMINI_FINISH_REASONS
    ):
        return True
    incomplete_details = response_metadata.get("incomplete_details")
    if (
        isinstance(incomplete_details, dict)
        and incomplete_details.get("reason") == "content_filter"
    ):
        return True
    prompt_feedback = response_metadata.get("prompt_feedback")
    if isinstance(prompt_feedback, dict):
        block_reason = prompt_feedback.get("block_reason")
        return (
            isinstance(block_reason, str) and block_reason in _GEMINI_BLOCK_REASONS
        ) or (type(block_reason) is int and block_reason in {1, 2, 3, 4})
    return False
