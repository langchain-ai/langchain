import datetime
import warnings
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.utils import check_package_version
from typing_extensions import TypedDict

try:
    check_package_version("lark", gte_version="1.1.5")
    from lark import Lark, Transformer, v_args

    _HAS_LARK = True
except ImportError:

    def v_args(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        """Dummy decorator for when lark is not installed."""
        return lambda _: None

    Transformer = object  # type: ignore[assignment,misc]
    Lark = object  # type: ignore[assignment,misc]
    _HAS_LARK = False

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    FilterDirective,
    Operation,
    Operator,
)

GRAMMAR = r"""
    ?program: func_call
    ?expr: func_call
        | value

    func_call: CNAME "(" [args] ")"

    ?value: SIGNED_INT -> int
        | SIGNED_FLOAT -> float
        | DATE -> date
        | DATETIME -> datetime
        | list
        | string
        | ("false" | "False" | "FALSE") -> false
        | ("true" | "True" | "TRUE") -> true

    args: expr ("," expr)*
    DATE.2: /["']?(\d{4}-[01]\d-[0-3]\d)["']?/
    DATETIME.2: /["']?\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d[Zz]?["']?/
    string: /'[^']*'/ | ESCAPED_STRING
    list: "[" [args] "]"

    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_FLOAT
    %import common.SIGNED_INT
    %import common.WS
    %ignore WS
"""


class ISO8601Date(TypedDict):
    """A date in ISO 8601 format (YYYY-MM-DD)."""

    date: str
    type: Literal["date"]


class ISO8601DateTime(TypedDict):
    """A datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)."""

    datetime: str
    type: Literal["datetime"]


@v_args(inline=True)
class QueryTransformer(Transformer):
    """Transform a query string into an intermediate representation."""

    def __init__(
        self,
        *args: Any,
        allowed_comparators: Sequence[Comparator] | None = None,
        allowed_operators: Sequence[Operator] | None = None,
        allowed_attributes: Sequence[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize the QueryTransformer.

        Args:
            *args: Positional arguments.
            allowed_comparators: Optional sequence of allowed comparators.
            allowed_operators: Optional sequence of allowed operators.
            allowed_attributes: Optional sequence of allowed attributes for comparators.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.allowed_comparators = allowed_comparators
        self.allowed_operators = allowed_operators
        self.allowed_attributes = allowed_attributes

    def program(self, *items: Any) -> tuple:
        """Transform the items into a tuple."""
        return items

    def func_call(self, func_name: Any, args: list) -> FilterDirective:
        """Transform a function name and args into a FilterDirective.

        Args:
            func_name: The name of the function.
            args: The arguments passed to the function.

        Returns:
            The filter directive.

        Raises:
            ValueError: If the function is a comparator and the first arg is not in the
            allowed attributes.
        """
        func = self._match_func_name(str(func_name))
        if isinstance(func, Comparator):
            if self.allowed_attributes and args[0] not in self.allowed_attributes:
                msg = (
                    f"Received invalid attributes {args[0]}. Allowed attributes are "
                    f"{self.allowed_attributes}"
                )
                raise ValueError(msg)
            return Comparison(comparator=func, attribute=args[0], value=args[1])
        if len(args) == 1 and func in (Operator.AND, Operator.OR):
            return args[0]
        return Operation(operator=func, arguments=args)

    def _match_func_name(self, func_name: str) -> Operator | Comparator:
        if func_name in set(Comparator):
            if (
                self.allowed_comparators is not None
                and func_name not in self.allowed_comparators
            ):
                msg = (
                    f"Received disallowed comparator {func_name}. Allowed "
                    f"comparators are {self.allowed_comparators}"
                )
                raise ValueError(msg)
            return Comparator(func_name)
        if func_name in set(Operator):
            if (
                self.allowed_operators is not None
                and func_name not in self.allowed_operators
            ):
                msg = (
                    f"Received disallowed operator {func_name}. Allowed operators"
                    f" are {self.allowed_operators}"
                )
                raise ValueError(msg)
            return Operator(func_name)
        msg = (
            f"Received unrecognized function {func_name}. Valid functions are "
            f"{list(Operator) + list(Comparator)}"
        )
        raise ValueError(msg)

    def args(self, *items: Any) -> tuple:
        """Transforms items into a tuple.

        Args:
            items: The items to transform.
        """
        return items

    def false(self) -> bool:
        """Returns false."""
        return False

    def true(self) -> bool:
        """Returns true."""
        return True

    def list(self, item: Any) -> list:
        """Transforms an item into a list.

        Args:
            item: The item to transform.
        """
        if item is None:
            return []
        return list(item)

    def int(self, item: Any) -> int:
        """Transforms an item into an int.

        Args:
            item: The item to transform.
        """
        return int(item)

    def float(self, item: Any) -> float:
        """Transforms an item into a float.

        Args:
            item: The item to transform.
        """
        return float(item)

    def date(self, item: Any) -> ISO8601Date:
        """Transforms an item into a ISO8601Date object.

        Args:
            item: The item to transform.

        Raises:
            ValueError: If the item is not in ISO 8601 date format.
        """
        item = str(item).strip("\"'")
        try:
            datetime.datetime.strptime(item, "%Y-%m-%d")  # noqa: DTZ007
        except ValueError:
            warnings.warn(
                "Dates are expected to be provided in ISO 8601 date format "
                "(YYYY-MM-DD).",
                stacklevel=3,
            )
        return {"date": item, "type": "date"}

    def datetime(self, item: Any) -> ISO8601DateTime:
        """Transforms an item into a ISO8601DateTime object.

        Args:
            item: The item to transform.

        Raises:
            ValueError: If the item is not in ISO 8601 datetime format.
        """
        item = str(item).strip("\"'")
        try:
            # Parse full ISO 8601 datetime format
            datetime.datetime.strptime(item, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            try:
                datetime.datetime.strptime(item, "%Y-%m-%dT%H:%M:%S")  # noqa: DTZ007
            except ValueError as e:
                msg = "Datetime values are expected to be in ISO 8601 format."
                raise ValueError(msg) from e
        return {"datetime": item, "type": "datetime"}

    def string(self, item: Any) -> str:
        """Transforms an item into a string.

        Removes escaped quotes.

        Args:
            item: The item to transform.
        """
        return str(item).strip("\"'")


def get_parser(
    allowed_comparators: Sequence[Comparator] | None = None,
    allowed_operators: Sequence[Operator] | None = None,
    allowed_attributes: Sequence[str] | None = None,
) -> Lark:
    """Return a parser for the query language.

    Args:
        allowed_comparators: The allowed comparators.
        allowed_operators: The allowed operators.
        allowed_attributes: The allowed attributes.

    Returns:
        Lark parser for the query language.
    """
    if not _HAS_LARK:
        msg = "Cannot import lark, please install it with 'pip install lark'."
        raise ImportError(msg)
    transformer = QueryTransformer(
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        allowed_attributes=allowed_attributes,
    )
    return Lark(GRAMMAR, parser="lalr", transformer=transformer, start="program")
