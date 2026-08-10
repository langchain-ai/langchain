"""TypedDicts with deferred string annotations for regression testing.

Kept in a separate module so `from __future__ import annotations` applies and
the annotations remain unevaluated at class creation time.
"""
from __future__ import annotations

from typing_extensions import NotRequired, Required, TypedDict


class PartialPayloadDeferred(TypedDict, total=False):
    required_value: Required[int]
    optional_value: str
    explicit_optional_value: NotRequired[bool]


class FullPayloadDeferred(TypedDict):
    required_value: int
    optional_value: NotRequired[str]
