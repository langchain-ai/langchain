from enum import Enum
from functools import wraps
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Union

from langchain.utilities.redis import TokenEscaper

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
        self, val: Any, val_type: type, operator: RedisFilterOperator
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
    def wrapper(instance: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
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
    """A RedisFilterField representing a tag in a Redis index."""

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

    def __init__(self, field: str):
        """Create a RedisTag FilterField

        Args:
            field (str): The name of the RedisTag field in the index to be queried
                against.
        """
        super().__init__(field)

    def _set_tag_value(
        self, other: Union[List[str], str], operator: RedisFilterOperator
    ) -> None:
        if isinstance(other, list):
            if not all(isinstance(tag, str) for tag in other):
                raise ValueError("All tags must be strings")
        else:
            other = [other]
        self._set_value(other, list, operator)

    @check_operator_misuse
    def __eq__(self, other: Union[List[str], str]) -> "RedisFilterExpression":
        """Create a RedisTag equality filter expression

        Args:
            other (Union[List[str], str]): The tag(s) to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisTag
            >>> filter = RedisTag("brand") == "nike"
        """
        self._set_tag_value(other, RedisFilterOperator.EQ)
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: Union[List[str], str]) -> "RedisFilterExpression":
        """Create a RedisTag inequality filter expression

        Args:
            other (Union[List[str], str]): The tag(s) to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisTag
            >>> filter = RedisTag("brand") != "nike"
        """
        self._set_tag_value(other, RedisFilterOperator.NE)
        return RedisFilterExpression(str(self))

    @property
    def _formatted_tag_value(self) -> str:
        return "|".join([self.escaper.escape(tag) for tag in self._value])

    def __str__(self) -> str:
        if not self._value:
            raise ValueError(
                f"Operator must be used before calling __str__. Operators are "
                f"{self.OPERATORS.values()}"
            )
        """Return the Redis Query syntax for a RedisTag filter expression"""
        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._formatted_tag_value,
        )


class RedisNum(RedisFilterField):
    """A RedisFilterField representing a numeric field in a Redis index."""

    OPERATORS: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: "==",
        RedisFilterOperator.NE: "!=",
        RedisFilterOperator.LT: "<",
        RedisFilterOperator.GT: ">",
        RedisFilterOperator.LE: "<=",
        RedisFilterOperator.GE: ">=",
    }
    OPERATOR_MAP: Dict[RedisFilterOperator, str] = {
        RedisFilterOperator.EQ: "@%s:[%f %f]",
        RedisFilterOperator.NE: "(-@%s:[%f %f])",
        RedisFilterOperator.GT: "@%s:[(%f +inf]",
        RedisFilterOperator.LT: "@%s:[-inf (%f]",
        RedisFilterOperator.GE: "@%s:[%f +inf]",
        RedisFilterOperator.LE: "@%s:[-inf %f]",
    }

    def __str__(self) -> str:
        """Return the Redis Query syntax for a Numeric filter expression"""
        if not self._value:
            raise ValueError(
                f"Operator must be used before calling __str__. Operators are "
                f"{self.OPERATORS.values()}"
            )

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
        """Create a Numeric equality filter expression

        Args:
            other (Number): The value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisNum
            >>> filter = RedisNum("zipcode") == 90210
        """
        self._set_value(other, Number, RedisFilterOperator.EQ)
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric inequality filter expression

        Args:
            other (Number): The value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisNum
            >>> filter = RedisNum("zipcode") != 90210
        """
        self._set_value(other, Number, RedisFilterOperator.NE)
        return RedisFilterExpression(str(self))

    def __gt__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a RedisNumeric greater than filter expression

        Args:
            other (Number): The value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") > 18
        """
        self._set_value(other, Number, RedisFilterOperator.GT)
        return RedisFilterExpression(str(self))

    def __lt__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric less than filter expression

        Args:
            other (Number): The value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") < 18
        """
        self._set_value(other, Number, RedisFilterOperator.LT)
        return RedisFilterExpression(str(self))

    def __ge__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric greater than or equal to filter expression

        Args:
            other (Number): The value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") >= 18
        """
        self._set_value(other, Number, RedisFilterOperator.GE)
        return RedisFilterExpression(str(self))

    def __le__(self, other: Union[int, float]) -> "RedisFilterExpression":
        """Create a Numeric less than or equal to filter expression

        Args:
            other (Number): The value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") <= 18
        """
        self._set_value(other, Number, RedisFilterOperator.LE)
        return RedisFilterExpression(str(self))


class RedisText(RedisFilterField):
    """A RedisFilterField representing a text field in a Redis index."""

    OPERATORS = {
        RedisFilterOperator.EQ: "==",
        RedisFilterOperator.NE: "!=",
        RedisFilterOperator.LIKE: "%",
    }
    OPERATOR_MAP = {
        RedisFilterOperator.EQ: '@%s:"%s"',
        RedisFilterOperator.NE: '(-@%s:"%s")',
        RedisFilterOperator.LIKE: "@%s:%s",
    }

    @check_operator_misuse
    def __eq__(self, other: str) -> "RedisFilterExpression":
        """Create a RedisText equality filter expression

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisText
            >>> filter = RedisText("job") == "engineer"
        """
        self._set_value(other, str, RedisFilterOperator.EQ)
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: str) -> "RedisFilterExpression":
        """Create a RedisText inequality filter expression

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisText
            >>> filter = RedisText("job") != "engineer"
        """
        self._set_value(other, str, RedisFilterOperator.NE)
        return RedisFilterExpression(str(self))

    def __mod__(self, other: str) -> "RedisFilterExpression":
        """Create a RedisText like filter expression

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain.vectorstores.redis import RedisText
            >>> filter = RedisText("job") % "engineer"
        """
        self._set_value(other, str, RedisFilterOperator.LIKE)
        return RedisFilterExpression(str(self))

    def __str__(self) -> str:
        if not self._value:
            raise ValueError(
                f"Operator must be used before calling __str__. Operators are "
                f"{self.OPERATORS.values()}"
            )

        try:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)
        except KeyError:
            raise Exception("Invalid operator")


class RedisFilterExpression:
    """A logical expression of RedisFilterFields.

    RedisFilterExpressions can be combined using the & and | operators to create
    complex logical expressions that evaluate to the Redis Query language.

    This presents an interface by which users can create complex queries
    without having to know the Redis Query language.

    Filter expressions are not initialized directly. Instead they are built
    by combining RedisFilterFields using the & and | operators.

    Examples:

        >>> from langchain.vectorstores.redis import RedisTag, RedisNum
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

    def __str__(self) -> str:
        # top level check that allows recursive calls to __str__
        if not self._filter and not self._operator:
            raise ValueError("Improperly initialized RedisFilterExpression")

        # allow for single filter expression without operators as last
        # expression in the chain might not have an operator
        if self._operator:
            operator_str = " | " if self._operator == RedisFilterOperator.OR else " "
            return f"({str(self._left)}{operator_str}{str(self._right)})"

        # check that base case, the filter is set
        if not self._filter:
            raise ValueError("Improperly initialized RedisFilterExpression")
        return self._filter
