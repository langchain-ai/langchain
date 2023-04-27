from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Sequence, Union

from pydantic import BaseModel


class Visitor(ABC):
    """Abstract visitor interface."""

    allowed_comparators: Optional[Sequence[Comparator]] = None
    allowed_operators: Optional[Sequence[Operator]] = None

    @abstractmethod
    def visit_operation(self, operation: Operation) -> Any:
        """"""

    @abstractmethod
    def visit_comparison(self, comparison: Comparison) -> Any:
        """"""

    @abstractmethod
    def visit_structured_query(self, structured_query: StructuredQuery) -> Any:
        """"""


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


class Comparison(Expr):
    comparator: Comparator
    attribute: str
    value: Any


class Operation(Expr):
    operator: Operator
    arguments: List[Union[Comparison, Operation]]


class StructuredQuery(Expr):
    query: str
    filter: Optional[Union[Comparison, Operation]]
