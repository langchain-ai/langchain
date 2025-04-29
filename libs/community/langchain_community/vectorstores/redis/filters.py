from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from langchain_community.utilities.redis import TokenEscaper

# disable mypy error for dunder method overrides
# mypy: disable-error-code="override"


class RedisFilterOperator(Enum):
    """RedisFilterOperator enumerator is used to create RedisFilterExpressions."""

    EQ = 1
    NE = 2
    LT = 3
    GT = 4
    LE = 5
    GE = 6
    OR = 7
    AND = 8
    LIKE = 9
    IN = 10


class RedisFilter:
    """Collection of RedisFilterFields."""

    @staticmethod
    def text(field: str) -> "RedisText":
        return RedisText(field)

    @staticmethod
    def num(field: str) -> "RedisNum":
        return RedisNum(field)

    @staticmethod
    def tag(field: str) -> "RedisTag":
        return RedisTag(field)


class RedisFilterField:
    """Base class for RedisFilterFields."""

    escaper: "TokenEscaper" = TokenEscaper()
    OPERATORS: Dict[RedisFilterOperator, str] = {}

    def __init__(self, field: str):
        self._field = field
        self._value: Any = None
        self._operator: RedisFilterOperator = RedisFilterOperator.EQ

    def equals(self, other: "RedisFilterField") -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._field == other._field and self._value == other._value

    def _set_value(
        self, val: Any, val_type: Tuple[Any], operator: RedisFilterOperator
    ) -> None:
        # check that the operator is supported by this class
        if operator not in self.OPERATORS:
            raise ValueError(
                f"Operator {operator} not supported by {self.__class__.__name__}. "
                + f"Supported operators are {self.OPERATORS.values()}."
            )

        if not isinstance(val, val_type):
            raise TypeError(
                f"Right side argument passed to operator {self.OPERATORS[operator]} "
                f"with left side "
                f"argument {self.__class__.__name__} must be of type {val_type}, "
                f"received value {val}"
            )
        self._value = val
        self._operator = operator


def check_operator_misuse(func: Callable) -> Callable:
    """Decorator to check for misuse of equality operators."""

    @wraps(func)
    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        # Extracting 'other' from positional arguments or keyword arguments
        other = kwargs.get("other") if "other" in kwargs else None
        if not other:
            for arg in args:
                if isinstance(arg, type(instance)):
                    other = arg
                    break

        if isinstance(other, type(instance)):
            raise ValueError(
                "Equality operators are overridden for FilterExpression creation. Use "
                ".equals() for equality checks"
            )
        return func(instance, *args, **kwargs)

    return wrapper


class RedisTag(RedisFilterField):
    """RedisFilterField representing a tag in a Redis index."""

    OPERATORS: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: "==",
        RedisFilterOperator.NE: "!=",
        RedisFilterOperator.IN: "==",
    }
    OPERATOR_MAP: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: "@%s:{%s}",
        RedisFilterOperator.NE: "(-@%s:{%s})",
        RedisFilterOperator.IN: "@%s:{%s}",
    }
    SUPPORTED_VAL_TYPES = (list, set, tuple, str, type(None))

    def __init__(self, field: str):
        """Create a RedisTag FilterField.

        Args:
            field (str): The name of the RedisTag field in the index to be queried
                against.
        """
        super().__init__(field)

    def _set_tag_value(
        self,
        other: Union[List[str], Set[str], Tuple[str], str],
        operator: RedisFilterOperator,
    ) -> None:
        if isinstance(other, (list, set, tuple)):
            try:
                # "if val" clause removes non-truthy values from list
                other = [str(val) for val in other if val]
            except ValueError:
                raise ValueError("All tags within collection must be strings")
        # above to catch the "" case
        elif not other:
            other = []
        elif isinstance(other, str):
            other = [other]

        self._set_value(other, self.SUPPORTED_VAL_TYPES, operator)  # type: ignore[arg-type]

    @check_operator_misuse
    def __eq__(
        self, other: Union[List[str], Set[str], Tuple[str], str]
    ) -> "RedisFilterExpression":
        """Create a RedisTag equality filter expression.

        Args:
            other (Union[List[str], Set[str], Tuple[str], str]):
                The tag(s) to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisTag
            >>> filter = RedisTag("brand") == "nike"
        """
        self._set_tag_value(other, RedisFilterOperator.EQ)
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(
        self, other: Union[List[str], Set[str], Tuple[str], str]
    ) -> "RedisFilterExpression":
        """Create a RedisTag inequality filter expression.

        Args:
            other (Union[List[str], Set[str], Tuple[str], str]):
                The tag(s) to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisTag
            >>> filter = RedisTag("brand") != "nike"
        """
        self._set_tag_value(other, RedisFilterOperator.NE)
        return RedisFilterExpression(str(self))

    @property
    def _formatted_tag_value(self) -> str:
        return "|".join([self.escaper.escape(tag) for tag in self._value])

    def __str__(self) -> str:
        """Return the query syntax for a RedisTag filter expression."""
        if not self._value:
            return "*"

        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._formatted_tag_value,
        )


class RedisNum(RedisFilterField):
    """RedisFilterField representing a numeric field in a Redis index."""

    OPERATORS: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: "==",
        RedisFilterOperator.NE: "!=",
        RedisFilterOperator.LT: "<",
        RedisFilterOperator.GT: ">",
        RedisFilterOperator.LE: "<=",
        RedisFilterOperator.GE: ">=",
    }
    OPERATOR_MAP: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: "@%s:[%s %s]",
        RedisFilterOperator.NE: "(-@%s:[%s %s])",
        RedisFilterOperator.GT: "@%s:[(%s +inf]",
        RedisFilterOperator.LT: "@%s:[-inf (%s]",
        RedisFilterOperator.GE: "@%s:[%s +inf]",
        RedisFilterOperator.LE: "@%s:[-inf %s]",
    }
    SUPPORTED_VAL_TYPES = (int, float, type(None))

    def __str__(self) -> str:
        """Return the query syntax for a RedisNum filter expression."""
        if self._value is None:
            return "*"

        if (
            self._operator == RedisFilterOperator.EQ
            or self._operator == RedisFilterOperator.NE
        ):
            return self.OPERATOR_MAP[self._operator] % (
                self._field,
                self._value,
                self._value,
            )
        else:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)

    @check_operator_misuse
    def __eq__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric equality filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("zipcode") == 90210
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.EQ)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric inequality filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("zipcode") != 90210
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.NE)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    def __gt__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric greater than filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") > 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.GT)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    def __lt__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric less than filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") < 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.LT)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    def __ge__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric greater than or equal to filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") >= 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.GE)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    def __le__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric less than or equal to filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") <= 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.LE)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))


class RedisText(RedisFilterField):
    """RedisFilterField representing a text field in a Redis index."""

    OPERATORS: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: "==",
        RedisFilterOperator.NE: "!=",
        RedisFilterOperator.LIKE: "%",
    }
    OPERATOR_MAP: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: '@%s:("%s")',
        RedisFilterOperator.NE: '(-@%s:"%s")',
        RedisFilterOperator.LIKE: "@%s:(%s)",
    }
    SUPPORTED_VAL_TYPES = (str, type(None))

    @check_operator_misuse
    def __eq__(self, other: str) -> "RedisFilterExpression":
        """Create a RedisText equality (exact match) filter expression.

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisText
            >>> filter = RedisText("job") == "engineer"
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.EQ)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: str) -> "RedisFilterExpression":
        """Create a RedisText inequality filter expression.

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisText
            >>> filter = RedisText("job") != "engineer"
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.NE)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    def __mod__(self, other: str) -> "RedisFilterExpression":
        """Create a RedisText "LIKE" filter expression.

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisText
            >>> filter = RedisText("job") % "engine*"         # suffix wild card match
            >>> filter = RedisText("job") % "%%engine%%"      # fuzzy match w/ LD
            >>> filter = RedisText("job") % "engineer|doctor" # contains either term
            >>> filter = RedisText("job") % "engineer doctor" # contains both terms
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.LIKE)  # type: ignore[arg-type]
        return RedisFilterExpression(str(self))

    def __str__(self) -> str:
        """Return the query syntax for a RedisText filter expression."""
        if not self._value:
            return "*"

        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._value,
        )


class RedisFilterExpression:
    """Logical expression of RedisFilterFields.

    RedisFilterExpressions can be combined using the & and | operators to create
    complex logical expressions that evaluate to the Redis Query language.

    This presents an interface by which users can create complex queries
    without having to know the Redis Query language.

    Filter expressions are not initialized directly. Instead they are built
    by combining RedisFilterFields using the & and | operators.

    Examples:

        >>> from langchain_community.vectorstores.redis import RedisTag, RedisNum
        >>> brand_is_nike = RedisTag("brand") == "nike"
        >>> price_is_under_100 = RedisNum("price") < 100
        >>> filter = brand_is_nike & price_is_under_100
        >>> print(str(filter))
        (@brand:{nike} @price:[-inf (100)])

    """

    def __init__(
        self,
        _filter: Optional[str] = None,
        operator: Optional[RedisFilterOperator] = None,
        left: Optional["RedisFilterExpression"] = None,
        right: Optional["RedisFilterExpression"] = None,
    ):
        self._filter = _filter
        self._operator = operator
        self._left = left
        self._right = right

    def __and__(self, other: "RedisFilterExpression") -> "RedisFilterExpression":
        return RedisFilterExpression(
            operator=RedisFilterOperator.AND, left=self, right=other
        )

    def __or__(self, other: "RedisFilterExpression") -> "RedisFilterExpression":
        return RedisFilterExpression(
            operator=RedisFilterOperator.OR, left=self, right=other
        )

    @staticmethod
    def format_expression(
        left: "RedisFilterExpression", right: "RedisFilterExpression", operator_str: str
    ) -> str:
        _left, _right = str(left), str(right)
        if _left == _right == "*":
            return _left
        if _left == "*" != _right:
            return _right
        if _right == "*" != _left:
            return _left
        return f"({_left}{operator_str}{_right})"

    def __str__(self) -> str:
        # top level check that allows recursive calls to __str__
        if not self._filter and not self._operator:
            raise ValueError("Improperly initialized RedisFilterExpression")

        # if there's an operator, combine expressions accordingly
        if self._operator:
            if not isinstance(self._left, RedisFilterExpression) or not isinstance(
                self._right, RedisFilterExpression
            ):
                raise TypeError(
                    "Improper combination of filters."
                    "Both left and right should be type FilterExpression"
                )

            operator_str = " | " if self._operator == RedisFilterOperator.OR else " "
            return self.format_expression(self._left, self._right, operator_str)

        # check that base case, the filter is set
        if not self._filter:
            raise ValueError("Improperly initialized RedisFilterExpression")
        return self._filter
