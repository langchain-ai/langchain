# SPDX-License-Identifier: MIT
"""Utility functions.

Parsing non-standard reasoning or thinking fields
from OpenAI-compatible chat completion responses (e.g., Qwen models).
"""

from __future__ import annotations

from typing import Any


def extract_reasoning_content(
    model_name: str, response_dict: dict[str, Any]
) -> str | None:
    """Extract 'reasoning_content' fields from an OpenAI-compatible response.

    This function handles Qwen-family models that provide internal reasoning or
    "think" traces in their message objects.
    """
    if not isinstance(response_dict, dict):
        return None

    choices = response_dict.get("choices")
    if not choices or not isinstance(choices, list):
        return None

    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return None

    if "qwen" in (model_name or "").lower():
        if "reasoning_content" in msg:
            return msg["reasoning_content"]
        for alt_key in ("think", "thought", "reasoning"):
            if alt_key in msg:
                return msg[alt_key]
    return None


def extract_reasoning_delta(model_name: str, delta_dict: dict[str, Any]) -> str | None:
    """Extract reasoning field from incremental streaming deltas for Qwen models.

    Used when consuming stream data (`choices[0].delta`)
    to combine partial reasoning text.
    """
    if not isinstance(delta_dict, dict):
        return None

    if "qwen" in (model_name or "").lower():
        for key in ("reasoning_content", "think", "thought"):
            if key in delta_dict:
                return delta_dict[key]
    return None
