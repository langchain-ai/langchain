"""Utilities for normalizing `tool_choice` across providers.

This provides a single place to normalize user-facing `tool_choice` values
into provider-specific representations (or canonical forms) so adapters can
consume them consistently.
"""
from __future__ import annotations

from typing import Any


def normalize_tool_choice(tool_choice: Any) -> Any:
    """Normalize common `tool_choice` inputs to canonical values.

    Normalizations applied:
    - `"any"` -> `"required"` (many providers use `required`)
    - `True` -> `"required"`
    - truthy strings that match known well-known tools are left as-is
    - `None` / `False` left as-is

    This function intentionally performs only conservative mappings; provider
    adapters may further map the result to provider-specific payloads.

    Args:
        tool_choice: Arbitrary user-supplied tool_choice value.

    Returns:
        The normalized value.
    """
    if tool_choice is None:
        return None
    # Booleans: True means require a tool, False means no-op
    if isinstance(tool_choice, bool):
        return "required" if tool_choice else None
    # Strings
    if isinstance(tool_choice, str):
        lc = tool_choice.lower()
        if lc == "any":
            return "required"
        # preserve 'auto', 'none', 'required', and specific names
        return tool_choice
    # dicts and other types: return as-is (adapters can validate)
    return tool_choice
