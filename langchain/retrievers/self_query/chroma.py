"""Logic for converting internal query language to a valid Chroma query."""
from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class ChromaTranslator(Visitor):
    """Logic for converting internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        if isinstance(func, Operator) and self.allowed_operators is not None:
            if func not in self.allowed_operators:
                raise ValueError(
                    f"Received disallowed operator {func}. Allowed "
                    f"comparators are {self.allowed_operators}"
                )
        if isinstance(func, Comparator) and self.allowed_comparators is not None:
            if func not in self.allowed_comparators:
                raise ValueError(
                    f"Received disallowed comparator {func}. Allowed "
                    f"comparators are {self.allowed_comparators}"
                )
        return f"${func.value}"

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {self._format_func(operation.operator): args}

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
