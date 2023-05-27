"""Logic for converting internal query language to a valid Weaviate query."""
from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class WeaviateTranslator(Visitor):
    """Logic for converting internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""

    allowed_comparators = [Comparator.EQ]

    def _map_func(self, func: Union[Operator, Comparator]) -> str:
        # https://weaviate.io/developers/weaviate/api/graphql/filters
        map_dict = {Operator.AND: "And", Operator.OR: "Or", Comparator.EQ: "Equal"}
        return map_dict[func]

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
        return self._map_func(func)

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {"operator": self._format_func(operation.operator), "operands": args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return {
            "path": [comparison.attribute],
            "operator": self._format_func(comparison.comparator),
            "valueText": comparison.value,
        }

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"where_filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
