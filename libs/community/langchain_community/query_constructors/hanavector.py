# HANA Translator/query constructor
from typing import Dict, Tuple, Union

from langchain_core._api import deprecated
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


@deprecated(
    since="0.3.23",
    removal="1.0",
    message=(
        "This class is deprecated and will be removed in a future version. "
        "Please use query_constructors.HanaTranslator from the "
        "langchain_hana package instead. "
        "See https://github.com/SAP/langchain-integration-for-sap-hana-cloud "
        "for details."
    ),
    alternative="from langchain_hana.query_constructors import HanaTranslator;",
    pending=False,
)
class HanaTranslator(Visitor):
    """
    **DEPRECATED**: This class is deprecated and will no longer be maintained.
    Please use query_constructors.HanaTranslator from the langchain_hana
    package instead. It offers an improved implementation and full support.

    Translate internal query language elements to valid filters params for
    HANA vectorstore.
    """

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.LT,
        Comparator.GTE,
        Comparator.LTE,
        Comparator.IN,
        Comparator.NIN,
        # Comparator.CONTAIN,
        Comparator.LIKE,
    ]

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
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
