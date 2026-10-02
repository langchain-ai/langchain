"""Opt-in, metadata-only diagnostics shared by conversation providers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

DEBUG_ENV_VAR = "LANGCHAIN_VOICE_DEBUG"
MAX_DEBUG_REFS = 8
MAX_DEBUG_REF_CHARS = 80


class DiagnosticUI(Protocol):
    """Minimal UI contract used for opt-in diagnostics."""

    def log(self, message: str) -> None:
        """Publish a diagnostic message."""
        ...


def debug_logging_enabled() -> bool:
    """Return whether metadata-only voice diagnostics are enabled."""
    value = os.getenv(DEBUG_ENV_VAR, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def make_debug_log(ui: DiagnosticUI) -> Callable[[str], None]:
    """Create a debug logger that respects the opt-in environment flag."""
    if not debug_logging_enabled():
        return lambda _message: None
    return lambda message: ui.log(f"[langchain.voice:debug] {message}")


def debug_refs(values: list[Any]) -> str:
    """Render bounded, single-line identifiers for diagnostic output."""
    refs = []
    for value in values[:MAX_DEBUG_REFS]:
        ref = str(value).replace("\r", "\\r").replace("\n", "\\n")
        refs.append(ref[:MAX_DEBUG_REF_CHARS])
    if len(values) > MAX_DEBUG_REFS:
        refs.append(f"+{len(values) - MAX_DEBUG_REFS}_more")
    return ",".join(refs) or "none"
