"""Schemas for tracers."""

from __future__ import annotations

from langsmith import RunTree

# Begin V2 API Schemas


Run = RunTree  # For backwards compatibility

__all__ = [
    "Run",
]
