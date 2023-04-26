from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Union

from pydantic import BaseModel

try:
    from lark import Lark, Transformer, v_args
except ImportError:
    pass


class Visitor(ABC):
    """Abstract visitor interface."""

    @abstractmethod
    def visit_operation(self, operation: "Operation") -> Any:
        """"""

    @abstractmethod
    def visit_comparison(self, comparison: "Comparison") -> Any:
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


class Expr(BaseModel, ABC):
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
    arguments: List[Union["Operation", Comparison]]


class StructuredQuery(BaseModel):
    query: str
    filter: Union[Comparison, Operation]
