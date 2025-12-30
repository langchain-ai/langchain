"""Compatibility helpers for Pydantic v1/v2 with langsmith Run objects."""

from __future__ import annotations

from typing import Any

from langchain_core.tracers.schemas import Run

# Detect Pydantic version once at import time based on Run model
_RUN_IS_PYDANTIC_V2 = hasattr(Run, "model_dump")


def run_to_dict(run: Run, **kwargs: Any) -> dict[str, Any]:
    """Convert run to dict, compatible with both Pydantic v1 and v2.

    Args:
        run: The run to convert.
        **kwargs: Additional arguments passed to model_dump/dict.

    Returns:
        Dictionary representation of the run.
    """
    if _RUN_IS_PYDANTIC_V2:
        return run.model_dump(**kwargs)
    return run.dict(**kwargs)  # type: ignore[deprecated]


def run_copy(run: Run, **kwargs: Any) -> Run:
    """Copy run, compatible with both Pydantic v1 and v2.

    Args:
        run: The run to copy.
        **kwargs: Additional arguments passed to model_copy/copy.

    Returns:
        A copy of the run.
    """
    if _RUN_IS_PYDANTIC_V2:
        return run.model_copy(**kwargs)
    return run.copy(**kwargs)  # type: ignore[deprecated]


def run_construct(**kwargs: Any) -> Run:
    """Construct run without validation, compatible with both Pydantic v1 and v2.

    Args:
        **kwargs: Fields to set on the run.

    Returns:
        A new Run instance constructed without validation.
    """
    if _RUN_IS_PYDANTIC_V2:
        return Run.model_construct(**kwargs)
    return Run.construct(**kwargs)  # type: ignore[deprecated]
