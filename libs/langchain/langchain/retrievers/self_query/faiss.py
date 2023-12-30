from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class FAISSTranslator(Visitor):
    """Translate `FAISS` Metadata filters to valid filters."""

    allowed_operators = [Operator.AND]
    """Subset of allowed logical operators."""
    allowed_comparators = [Comparator.EQ, Comparator.CONTAIN]
    """Subset of allowed logical comparators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return ""

    def visit_operation(self, operation: Operation) -> Dict:
        comparison_dict = {}
        for arg in operation.arguments:
            arg: Comparison
            if arg.attribute in comparison_dict:
                comparison_dict[arg.attribute].append(arg.accept(self))
            else:
                comparison_dict[arg.attribute] = [arg.accept(self)]
        return comparison_dict

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return comparison.value

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
