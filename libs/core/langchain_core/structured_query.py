"""Internal representation of a structured query language."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence


class Visitor(ABC):
    """Defines interface for IR translation using a visitor pattern."""

    allowed_comparators: Optional[Sequence[Comparator]] = None
    """Allowed comparators for the visitor."""
    allowed_operators: Optional[Sequence[Operator]] = None
    """Allowed operators for the visitor."""

    def _validate_func(self, func: Union[Operator, Comparator]) -> None:
        if (
            isinstance(func, Operator)
            and self.allowed_operators is not None
            and func not in self.allowed_operators
        ):
            msg = (
                f"Received disallowed operator {func}. Allowed "
                f"comparators are {self.allowed_operators}"
            )
            raise ValueError(msg)
        if (
            isinstance(func, Comparator)
            and self.allowed_comparators is not None
            and func not in self.allowed_comparators
        ):
            msg = (
                f"Received disallowed comparator {func}. Allowed "
                f"comparators are {self.allowed_comparators}"
            )
            raise ValueError(msg)

    @abstractmethod
    def visit_operation(self, operation: Operation) -> Any:
        """Translate an Operation.

        Args:
            operation: Operation to translate.
        """

    @abstractmethod
    def visit_comparison(self, comparison: Comparison) -> Any:
        """Translate a Comparison.

        Args:
            comparison: Comparison to translate.
        """

    @abstractmethod
    def visit_structured_query(self, structured_query: StructuredQuery) -> Any:
        """Translate a StructuredQuery.

        Args:
            structured_query: StructuredQuery to translate.
        """


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
            visitor: visitor to accept.

        Returns:
            result of visiting.
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
    """Filtering expression."""


class Comparison(FilterDirective):
    """Comparison to a value."""

    comparator: Comparator
    """The comparator to use."""
    attribute: str
    """The attribute to compare."""
    value: Any
    """The value to compare to."""

    def __init__(
        self, comparator: Comparator, attribute: str, value: Any, **kwargs: Any
    ) -> None:
        """Create a Comparison.

        Args:
            comparator: The comparator to use.
            attribute: The attribute to compare.
            value: The value to compare to.
        """
        # super exists from BaseModel
        super().__init__(  # type: ignore[call-arg]
            comparator=comparator, attribute=attribute, value=value, **kwargs
        )


class Operation(FilterDirective):
    """Logical operation over other directives."""

    operator: Operator
    """The operator to use."""
    arguments: list[FilterDirective]
    """The arguments to the operator."""

    def __init__(
        self, operator: Operator, arguments: list[FilterDirective], **kwargs: Any
    ) -> None:
        """Create an Operation.

        Args:
            operator: The operator to use.
            arguments: The arguments to the operator.
        """
        # super exists from BaseModel
        super().__init__(  # type: ignore[call-arg]
            operator=operator, arguments=arguments, **kwargs
        )


class StructuredQuery(Expr):
    """Structured query."""

    query: str
    """Query string."""
    filter: Optional[FilterDirective]
    """Filtering expression."""
    limit: Optional[int]
    """Limit on the number of results."""

    def __init__(
        self,
        query: str,
        filter: Optional[FilterDirective],  # noqa: A002
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Create a StructuredQuery.

        Args:
            query: The query string.
            filter: The filtering expression.
            limit: The limit on the number of results.
        """
        # super exists from BaseModel
        super().__init__(  # type: ignore[call-arg]
            query=query, filter=filter, limit=limit, **kwargs
        )
