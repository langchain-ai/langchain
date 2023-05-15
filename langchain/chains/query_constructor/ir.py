"""Internal representation of a structured query language."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Sequence

from pydantic import BaseModel


class Visitor(ABC):
    """Defines interface for IR translation using visitor pattern."""

    allowed_comparators: Optional[Sequence[Comparator]] = None
    allowed_operators: Optional[Sequence[Operator]] = None

    @abstractmethod
    def visit_operation(self, operation: Operation) -> Any:
        """Translate an Operation."""

    @abstractmethod
    def visit_comparison(self, comparison: Comparison) -> Any:
        """Translate a Comparison."""

    @abstractmethod
    def visit_structured_query(self, structured_query: StructuredQuery) -> Any:
        """Translate a StructuredQuery."""


def _to_snake_case(name: str) -> str:
    """Convert a name into snake_case."""
    snake_case = ""
    for i, char in enumerate(name):
        if char.isupper() and i != 0:
            snake_case += "_" + char.lower()
        else:
            snake_case += char.lower()
    return snake_case


class Expr(BaseModel):
    def accept(self, visitor: Visitor) -> Any:
        return getattr(visitor, f"visit_{_to_snake_case(self.__class__.__name__)}")(
            self
        )


class Operator(str, Enum):
    AND = "and"
    OR = "or"
    NOT = "not"


class Comparator(str, Enum):
    EQ = "eq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class FilterDirective(Expr, ABC):
    """A filtering expression."""


class Comparison(FilterDirective):
    """A comparison to a value."""

    comparator: Comparator
    attribute: str
    value: Any


class Operation(FilterDirective):
    """A logical operation over other directives."""

    operator: Operator
    arguments: List[FilterDirective]


class StructuredQuery(Expr):
    query: str
    filter: Optional[FilterDirective]
    limit: Optional[int]
