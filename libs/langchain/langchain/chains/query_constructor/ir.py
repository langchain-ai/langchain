"""Internal representation of a structured query language."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Sequence, Union

from langchain_core.pydantic_v1 import BaseModel


class Visitor(ABC):
    """Defines interface for IR translation using visitor pattern."""

    allowed_comparators: Optional[Sequence[Comparator]] = None
    allowed_operators: Optional[Sequence[Operator]] = None

    def _validate_func(self, func: Union[Operator, Comparator]) -> None:
        if isinstance(func, Operator) and self.allowed_operators is not None:
            if func not in self.allowed_operators:
                raise ValueError(
                    f"Received disallowed operator {func}. Allowed "
                    f"comparators are {self.allowed_operators}"
                )
        if isinstance(func, Comparator) and self.allowed_comparators is not None:
            if func not in self.allowed_comparators:
                raise ValueError(
                    f"Received disallowed comparator {func}. Allowed "
                    f"comparators are {self.allowed_comparators}"
                )

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
    """Base class for all expressions."""

    def accept(self, visitor: Visitor) -> Any:
        """Accept a visitor.

        Args:
            visitor: visitor to accept

        Returns:
            result of visiting
        """
        return getattr(visitor, f"visit_{_to_snake_case(self.__class__.__name__)}")(
            self
        )


class Operator(str, Enum):
    """Enumerator of the operations."""

    AND = "and"
    OR = "or"
    NOT = "not"


class Comparator(str, Enum):
    """Enumerator of the comparison operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAIN = "contain"
    LIKE = "like"
    IN = "in"
    NIN = "nin"


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
    """A structured query."""

    query: str
    """Query string."""
    filter: Optional[FilterDirective]
    """Filtering expression."""
    limit: Optional[int]
    """Limit on the number of results."""
