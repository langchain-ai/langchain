""""""
from typing import Dict, List, Tuple, Union

from langchain.chains.query_constructor.query_language import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class PineconeTranslator(Visitor):
    allowed_operators = [Operator.AND, Operator.OR]

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        """"""
        if isinstance(func, Operator) and self.allowed_operators is not None:
            if func not in self.allowed_operators:
                raise ValueError
        if isinstance(func, Comparator) and self.allowed_comparators is not None:
            if func not in self.allowed_comparators:
                raise ValueError
        return f"${func}"

    def visit_operation(self, operation: Operation) -> Dict:
        root: Dict = {self._format_func(operation.operator): []}
        stack: List[Tuple[Dict, List]] = [(root, operation.arguments)]
        while stack:
            op_dict, op_args = stack.pop()
            new_args: List = list(op_dict.values())[0]
            for arg in op_args:
                if isinstance(arg, Comparison):
                    new_args.append(self.visit_comparison(arg))
                else:
                    next_op: Dict = {self._format_func(arg.operator): []}
                    new_args.append(next_op)
                    stack.append((next_op, arg.arguments))
        return root

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return {
            comparison.attribute: {
                self._format_func(comparison.comparator): comparison.value
            }
        }

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
