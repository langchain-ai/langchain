import datetime
from typing import Any, Optional, Sequence, Union

try:
    import lark
    from packaging import version

    if version.parse(lark.__version__) < version.parse("1.1.5"):
        raise ValueError(
            f"Lark should be at least version 1.1.5, got {lark.__version__}"
        )
    from lark import Lark, Transformer, v_args
except ImportError:

    def v_args(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return lambda _: None

    Transformer = object  # type: ignore
    Lark = object  # type: ignore

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    FilterDirective,
    Operation,
    Operator,
)

GRAMMAR = """
    ?program: func_call
    ?expr: func_call
        | value

    func_call: CNAME "(" [args] ")"

    ?value: SIGNED_INT -> int
        | SIGNED_FLOAT -> float
        | TIMESTAMP -> timestamp
        | list
        | string
        | ("false" | "False" | "FALSE") -> false
        | ("true" | "True" | "TRUE") -> true

    args: expr ("," expr)*
    TIMESTAMP.2: /["'](\d{4}-[01]\d-[0-3]\d)["']/
    string: /'[^']*'/ | ESCAPED_STRING
    list: "[" [args] "]"

    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_FLOAT
    %import common.SIGNED_INT
    %import common.WS
    %ignore WS
"""


@v_args(inline=True)
class QueryTransformer(Transformer):
    def __init__(
        self,
        *args: Any,
        allowed_comparators: Optional[Sequence[Comparator]] = None,
        allowed_operators: Optional[Sequence[Operator]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.allowed_comparators = allowed_comparators
        self.allowed_operators = allowed_operators

    def program(self, *items: Any) -> tuple:
        return items

    def func_call(self, func_name: Any, args: list) -> FilterDirective:
        func = self._match_func_name(str(func_name))
        if isinstance(func, Comparator):
            return Comparison(comparator=func, attribute=args[0], value=args[1])
        elif len(args) == 1 and func in (Operator.AND, Operator.OR):
            return args[0]
        else:
            return Operation(operator=func, arguments=args)

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
        if item is None:
            return []
        return list(item)

    def int(self, item: Any) -> int:
        return int(item)

    def float(self, item: Any) -> float:
        return float(item)

    def timestamp(self, item: Any) -> datetime.date:
        item = item.replace("'", '"')
        return datetime.datetime.strptime(item, '"%Y-%m-%d"').date()

    def string(self, item: Any) -> str:
        # Remove escaped quotes
        return str(item).strip("\"'")


def get_parser(
    allowed_comparators: Optional[Sequence[Comparator]] = None,
    allowed_operators: Optional[Sequence[Operator]] = None,
) -> Lark:
    transformer = QueryTransformer(
        allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
    )
    return Lark(GRAMMAR, parser="lalr", transformer=transformer, start="program")
