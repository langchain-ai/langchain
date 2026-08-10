"""TypedDicts with deferred (string) annotations for regression testing.

Kept in a separate module (rather than inline in a test function) because
`from __future__ import annotations` applies to the whole file, and we need
these classes' annotations to genuinely be unevaluated strings at class
creation time to reproduce https://github.com/langchain-ai/langchain/pull/39336#discussion.
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
