"""Shared helpers for agent progress detection."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_MIN_CONSECUTIVE_STEPS = 2
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_REQUEST_ID_RE = re.compile(r"(?i)\b(?:request|req)(?:[_ -]?id)?\b\s*[:=]\s*[\w-]+")
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[tT ][0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
)
_WHITESPACE_RE = re.compile(r"\s+")


def stable_json_dumps(value: Any) -> str:
    """Serialize values deterministically for signature comparison."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def default_progress_output_normalizer(output: str) -> str:
    """Normalize volatile output details before signature comparison."""
    normalized = _TRACEBACK_RE.sub("", output)
    normalized = _UUID_RE.sub("<uuid>", normalized)
    normalized = _REQUEST_ID_RE.sub("request_id=<request_id>", normalized)
    normalized = _TIMESTAMP_RE.sub("<timestamp>", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def summarize_progress_output(output: str) -> str:
    """Prepare an output string for user-facing messages."""
    return _WHITESPACE_RE.sub(" ", output).strip()


def build_tool_failure_signature(
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    error_message: str,
    error_normalizer: Callable[[str], str],
) -> str:
    """Build a stable signature for a retry-exhausted tool failure."""
    return stable_json_dumps(
        {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "error": error_normalizer(error_message),
        }
    )


def build_tool_exchange_signature(
    *,
    tool_calls: Sequence[Mapping[str, Any]],
    tool_outputs: Sequence[Mapping[str, Any]],
) -> str:
    """Build a stable signature for a completed AI/tool exchange."""
    return stable_json_dumps(
        {
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs,
        }
    )


def validate_max_consecutive_steps(
    max_consecutive_steps: int, *, parameter_name: str
) -> None:
    """Validate a progress detection threshold."""
    if max_consecutive_steps < _MIN_CONSECUTIVE_STEPS:
        msg = f"{parameter_name} must be >= {_MIN_CONSECUTIVE_STEPS}"
        raise ValueError(msg)


def build_progress_stalled_message(*, consecutive_steps: int, description: str) -> str:
    """Build the final progress-stalled notification shown to users."""
    return (
        "Agent stopped because no_progress_detected: repeated the same tool "
        f"exchange {consecutive_steps} consecutive times. Last exchange: {description}"
    )


class AgentProgressStalledError(Exception):
    """Raised when an agent repeats equivalent tool exchanges without progress.

    The `reason` attribute is always `"no_progress_detected"` so callers can handle
    this stop condition without parsing the exception message.
    """

    reason = "no_progress_detected"

    def __init__(
        self,
        *,
        consecutive_steps: int,
        max_consecutive_identical_steps: int,
        description: str,
        exchange_signature: str | None = None,
    ) -> None:
        """Initialize the exception with progress stall details.

        Args:
            consecutive_steps: Number of equivalent tool exchanges observed.
            max_consecutive_identical_steps: Configured threshold that was reached.
            description: Concise description of the repeated exchange.
            exchange_signature: Stable signature for the repeated exchange, if available.
        """
        self.consecutive_steps = consecutive_steps
        self.max_consecutive_identical_steps = max_consecutive_identical_steps
        self.description = description
        self.exchange_signature = exchange_signature

        msg = build_progress_stalled_message(
            consecutive_steps=consecutive_steps,
            description=description,
        )
        super().__init__(msg)
