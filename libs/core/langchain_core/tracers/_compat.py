"""Compatibility helpers for Pydantic v1/v2 with langsmith `Run` objects.

!!! note

    The generic helpers (`pydantic_to_dict`, `pydantic_copy`) detect Pydanti version
    based on the langsmith `Run` model. They're intended for langsmith objects (`Run`,
    `Example`) which migrate together.

For general Pydantic v1/v2 handling, see `langchain_core.utils.pydantic`.
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.tracers.schemas import Run

# Detect Pydantic version once at import time based on Run model
_RUN_IS_PYDANTIC_V2 = hasattr(Run, "model_dump")

T = TypeVar("T")


def run_to_dict(run: Run, **kwargs: Any) -> dict[str, Any]:
    """Convert run to dict, compatible with both Pydantic v1 and v2.

    Args:
        run: The run to convert.
        **kwargs: Additional arguments passed to `model_dump`/`dict`.

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
        **kwargs: Additional arguments passed to `model_copy`/`copy`.

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
        A new `Run` instance constructed without validation.
    """
    if _RUN_IS_PYDANTIC_V2:
        return Run.model_construct(**kwargs)
    return Run.construct(**kwargs)  # type: ignore[deprecated]


def pydantic_to_dict(obj: Any, **kwargs: Any) -> dict[str, Any]:
    """Convert any Pydantic model to dict, compatible with both v1 and v2.

    Args:
        obj: The Pydantic model to convert.
        **kwargs: Additional arguments passed to `model_dump`/`dict`.

    Returns:
        Dictionary representation of the model.
    """
    if _RUN_IS_PYDANTIC_V2:
        return obj.model_dump(**kwargs)  # type: ignore[no-any-return]
    return obj.dict(**kwargs)  # type: ignore[no-any-return]


def pydantic_copy(obj: T, **kwargs: Any) -> T:
    """Copy any Pydantic model, compatible with both v1 and v2.

    Args:
        obj: The Pydantic model to copy.
        **kwargs: Additional arguments passed to `model_copy`/`copy`.

    Returns:
        A copy of the model.
    """
    if _RUN_IS_PYDANTIC_V2:
        return obj.model_copy(**kwargs)  # type: ignore[attr-defined,no-any-return]
    return obj.copy(**kwargs)  # type: ignore[attr-defined,no-any-return]
