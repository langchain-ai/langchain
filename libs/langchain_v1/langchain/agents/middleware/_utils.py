"""Utility functions for middleware."""

from typing import Any


def _generate_correction_tool_messages(content: str, tool_calls: list) -> list[dict[str, Any]]:
    """Generate tool messages for model behavior correction."""
    return [
        {"role": "tool", "content": content, "tool_call_id": tool_call["id"]}
        for tool_call in tool_calls
    ]
