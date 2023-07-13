import datetime
import re
from typing import Any, Callable, Dict, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


def DEFAULT_COMPOSER(op_name: str) -> Callable:
    """
    Default composer for logical operators.

    Args:
        op_name: Name of the operator.

    Returns:
        Callable that takes a list of arguments and returns a string.
    """

    def f(*args: Any) -> str:
        args_: map[str] = map(str, args)
        return f" {op_name} ".join(args_)

    return f


def FUNCTION_COMPOSER(op_name: str) -> Callable:
    """
    Composer for functions.
    Args:
        op_name: Name of the function.

    Returns:
        Callable that takes a list of arguments and returns a string.
    """

    def f(*args: Any) -> str:
        args_: map[str] = map(str, args)
        return f"{op_name}({','.join(args_)})"

    return f


class MyScaleTranslator(Visitor):
    """Logic for converting internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    """Subset of allowed logical operators."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.CONTAIN,
        Comparator.LIKE,
    ]

    map_dict = {
        Operator.AND: DEFAULT_COMPOSER("AND"),
        Operator.OR: DEFAULT_COMPOSER("OR"),
        Operator.NOT: DEFAULT_COMPOSER("NOT"),
        Comparator.EQ: DEFAULT_COMPOSER("="),
        Comparator.GT: DEFAULT_COMPOSER(">"),
        Comparator.GTE: DEFAULT_COMPOSER(">="),
        Comparator.LT: DEFAULT_COMPOSER("<"),
        Comparator.LTE: DEFAULT_COMPOSER("<="),
        Comparator.CONTAIN: FUNCTION_COMPOSER("has"),
        Comparator.LIKE: DEFAULT_COMPOSER("ILIKE"),
    }

    def __init__(self, metadata_key: str = "metadata") -> None:
        super().__init__()
        self.metadata_key = metadata_key

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        func = operation.operator
        self._validate_func(func)
        return self.map_dict[func](*args)

    def visit_comparison(self, comparison: Comparison) -> Dict:
        regex = "\((.*?)\)"
        matched = re.search("\(\w+\)", comparison.attribute)

        # If arbitrary function is applied to an attribute
        if matched:
            attr = re.sub(
                regex,
                f"({self.metadata_key}.{matched.group(0)[1:-1]})",
                comparison.attribute,
            )
        else:
            attr = f"{self.metadata_key}.{comparison.attribute}"
        value = comparison.value
        comp = comparison.comparator

        value = f"'{value}'" if type(value) is str else value

        # convert timestamp for datetime objects
        if type(value) is datetime.date:
            attr = f"parseDateTime32BestEffort({attr})"
            value = f"parseDateTime32BestEffort('{value.strftime('%Y-%m-%d')}')"

        # string pattern match
        if comp is Comparator.LIKE:
            value = f"'%{value[1:-1]}%'"
        return self.map_dict[comp](attr, value)

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        print(structured_query)
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"where_str": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
