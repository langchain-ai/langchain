"""Models with lazy string annotations for `_create_subset_model_v2` tests.

The `from __future__ import annotations` import is load-bearing: it makes every
raw value in this module's `__annotations__` an unresolved string, which is the
scenario the subset-model annotation tests exercise.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel


class FutureModel(BaseModel):
    """Model whose raw `__annotations__` values are unresolved strings."""

    metadata: dict[str, Any] | None = None
    tagged: Annotated[dict, "extra"] | None = None
