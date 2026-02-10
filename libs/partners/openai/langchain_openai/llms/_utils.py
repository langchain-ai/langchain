"""Internal utility functions for OpenAI LLM implementations."""

from __future__ import annotations

from typing import Any


def merge_stop_tokens(
    existing_stop: list[str] | str | None,
    new_stop: list[str] | str | None,
) -> list[str] | None:
    """Merge stop tokens from multiple sources, deduplicating while preserving order.

    This utility ensures that stop tokens from model_kwargs and runtime parameters
    are properly combined instead of one overwriting the other.

    Args:
        existing_stop: Existing stop tokens (from model_kwargs or params).
        new_stop: New stop tokens to merge (from runtime parameters).

    Returns:
        Merged and deduplicated list of stop tokens, or None if both inputs are None.

    Example:
        ```python
        # Merge model_kwargs stop with runtime stop
        result = merge_stop_tokens(["<|eot_id|>"], ["STOP", "END"])
        # Returns: ["<|eot_id|>", "STOP", "END"]

        # Handle duplicates
        result = merge_stop_tokens(["STOP"], ["STOP", "END"])
        # Returns: ["STOP", "END"]

        # Handle string inputs
        result = merge_stop_tokens("<|eot_id|>", "STOP")
        # Returns: ["<|eot_id|>", "STOP"]
        ```
    """
    if existing_stop is None and new_stop is None:
        return None

    # Convert inputs to lists
    existing_list: list[str] = []
    if existing_stop is not None:
        if isinstance(existing_stop, str):
            existing_list = [existing_stop]
        else:
            existing_list = list(existing_stop)

    new_list: list[str] = []
    if new_stop is not None:
        if isinstance(new_stop, str):
            new_list = [new_stop]
        else:
            new_list = list(new_stop)

    # If only one source has values, return it
    if not existing_list:
        return new_list if new_list else None
    if not new_list:
        return existing_list

    # Merge and deduplicate while preserving order
    # Use dict.fromkeys() to deduplicate while maintaining insertion order (Python 3.7+)
    merged = list(dict.fromkeys(existing_list + new_list))
    return merged


def extract_stop_tokens_from_params(params: dict[str, Any]) -> list[str] | None:
    """Extract and normalize stop tokens from a params dictionary.

    Args:
        params: Dictionary that may contain a "stop" key.

    Returns:
        List of stop tokens, or None if not present.
    """
    stop = params.get("stop")
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)
