from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel

try:
    from lark import Lark, Transformer, v_args
except ImportError:
    pass


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


class Comparison(BaseModel):
    comparator: Comparator
    attribute: str
    value: Any


class Operation(BaseModel):
    operator: Operator
    arguments: List[Union["Operation", Comparison]]


GRAMMAR = """
    ?program: func_call
    ?expr: func_call
        | value

    func_call: CNAME "(" [args] ")"

    ?value: SIGNED_NUMBER -> number
        | list
        | string
        | "false" -> false
        | "true" -> true

    args: expr ("," expr)*
    string: ESCAPED_STRING
    list: "[" [args] "]"

    %import common.CNAME
    %import common.SIGNED_NUMBER
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
"""


@v_args(inline=True)
class QueryTransformer(Transformer):
    def __init__(
        self,
        *args: Any,
        allowed_comparators: Optional[List[Comparator]] = None,
        allowed_operators: Optional[List[Operator]],
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.allowed_comparators = allowed_comparators
        self.allowed_operators = allowed_operators

    def program(self, *items: Any) -> tuple:
        return items

    def func_call(self, func_name: Any, *args: Any) -> Union[Comparison, Operation]:
        func = self._match_func_name(str(func_name))
        if isinstance(func, Comparator):
            return Comparison(comparator=func, attribute=args[0][0], value=args[0][1])
        return Operation(operator=func, arguments=args[0])

    def _match_func_name(self, func_name: str) -> Union[Operator, Comparator]:
        if func_name in set(Comparator):
            if self.allowed_comparators is not None:
                if func_name not in self.allowed_comparators:
                    raise ValueError(
                        f"Received disallowed comparator {func_name}. Allowed "
                        f"comparators are {self.allowed_comparators}"
                    )
            return Comparator(func_name)
        elif func_name in set(Operator):
            if self.allowed_operators is not None:
                if func_name not in self.allowed_operators:
                    raise ValueError(
                        f"Received disallowed operator {func_name}. Allowed operators"
                        f" are {self.allowed_operators}"
                    )
            return Operator(func_name)
        else:
            raise ValueError(
                f"Received unrecognized function {func_name}. Valid functions are "
                f"{list(Operator) + list(Comparator)}"
            )

    def args(self, *items: Any) -> tuple:
        return items

    def false(self) -> bool:
        return False

    def true(self) -> bool:
        return True

    def list(self, item: Any) -> list:
        return list(item)

    def number(self, item: Any) -> float:
        return float(item)

    def string(self, item: Any) -> str:
        # Remove escaped quotes
        return str(item).strip("\"'")


def get_parser(
    allowed_comparators: Optional[List[Comparator]] = None,
    allowed_operators: Optional[List[Operator]] = None,
) -> Lark:
    transformer = QueryTransformer(
        allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
    )
    return Lark(GRAMMAR, parser="lalr", transformer=transformer, start="program")
