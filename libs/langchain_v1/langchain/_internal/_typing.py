"""Private typing utilities for langchain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeAlias, TypeVar

from langgraph.graph._node import StateNode
from pydantic import BaseModel

if TYPE_CHECKING:
    from dataclasses import Field


class TypedDictLikeV1(Protocol):
    """Protocol to represent types that behave like ``TypedDict``s.

    Version 1: using ``ClassVar`` for keys.

    """

    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]


class TypedDictLikeV2(Protocol):
    """Protocol to represent types that behave like ``TypedDict``s.

    Version 2: not using ``ClassVar`` for keys.

    """

    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]


class DataclassLike(Protocol):
    """Protocol to represent types that behave like dataclasses.

    Inspired by the private ``_DataclassT`` from dataclasses that uses a similar
    protocol as a bound.

    """

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


StateLike: TypeAlias = TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel
"""Type alias for state-like types.

It can either be a ``TypedDict``, ``dataclass``, or Pydantic ``BaseModel``.

!!! note
    We cannot use either ``TypedDict`` or ``dataclass`` directly due to limitations in
    type checking.

"""

StateT = TypeVar("StateT", bound=StateLike)
"""Type variable used to represent the state in a graph."""

ContextT = TypeVar("ContextT", bound=StateLike | None)
"""Type variable for context types."""


__all__ = [
    "ContextT",
    "StateLike",
    "StateNode",
    "StateT",
]
