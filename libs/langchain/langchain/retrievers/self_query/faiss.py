from typing import Any, Dict, List, Tuple, Union

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
    allowed_comparators = [Comparator.EQ, Comparator.IN]
    """Subset of allowed logical comparators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return ""

    def visit_operation(self, operation: Operation) -> Dict:
        comparison_dict: Dict[str, Union[List, Any]] = {}
        arguments = operation.arguments
        for index in range(len(arguments)):
            arg = arguments[index]
            comparison_dict_instance: Dict = arg.accept(self)
            comparison_attribute, comparision_value = list(
                comparison_dict_instance.items()
            )[0]
            if comparison_attribute in comparison_dict:
                comparison_dict[comparison_attribute].append(comparision_value)
            else:
                comparison_dict[comparison_attribute] = [comparision_value]
        return comparison_dict

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return {comparison.attribute: comparison.value}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        print(kwargs)
        return structured_query.query, kwargs
