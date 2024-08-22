from __future__ import annotations

from typing import Optional, Sequence, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class TencentVectorDBTranslator(Visitor):
    """Translate StructuredQuery to Tencent VectorDB query."""

    COMPARATOR_MAP = {
        Comparator.EQ: "=",
        Comparator.NE: "!=",
        Comparator.GT: ">",
        Comparator.GTE: ">=",
        Comparator.LT: "<",
        Comparator.LTE: "<=",
        Comparator.IN: "in",
        Comparator.NIN: "not in",
    }

    allowed_comparators: Optional[Sequence[Comparator]] = list(COMPARATOR_MAP.keys())
    allowed_operators: Optional[Sequence[Operator]] = [
        Operator.AND,
        Operator.OR,
        Operator.NOT,
    ]

    def __init__(self, meta_keys: Optional[Sequence[str]] = None):
        """Initialize the translator.

        Args:
            meta_keys: List of meta keys to be used in the query. Default: [].
        """
        self.meta_keys = meta_keys or []

    def visit_operation(self, operation: Operation) -> str:
        """Visit an operation node and return the translated query.

        Args:
            operation: Operation node to be visited.

        Returns:
            Translated query.
        """
        if operation.operator in (Operator.AND, Operator.OR):
            ret = f" {operation.operator.value} ".join(
                [arg.accept(self) for arg in operation.arguments]
            )
            if operation.operator == Operator.OR:
                ret = f"({ret})"
            return ret
        else:
            return f"not ({operation.arguments[0].accept(self)})"

    def visit_comparison(self, comparison: Comparison) -> str:
        """Visit a comparison node and return the translated query.

        Args:
            comparison: Comparison node to be visited.

        Returns:
            Translated query.
        """
        if self.meta_keys and comparison.attribute not in self.meta_keys:
            raise ValueError(
                f"Expr Filtering found Unsupported attribute: {comparison.attribute}"
            )

        if comparison.comparator in self.COMPARATOR_MAP:
            if comparison.comparator in [Comparator.IN, Comparator.NIN]:
                value = map(
                    lambda x: f'"{x}"' if isinstance(x, str) else x, comparison.value
                )
                return (
                    f"{comparison.attribute}"
                    f" {self.COMPARATOR_MAP[comparison.comparator]} "
                    f"({', '.join(value)})"
                )
            if isinstance(comparison.value, str):
                return (
                    f"{comparison.attribute} "
                    f"{self.COMPARATOR_MAP[comparison.comparator]}"
                    f' "{comparison.value}"'
                )
            return (
                f"{comparison.attribute}"
                f" {self.COMPARATOR_MAP[comparison.comparator]} "
                f"{comparison.value}"
            )
        else:
            raise ValueError(f"Unsupported comparator {comparison.comparator}")

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        """Visit a structured query node and return the translated query.

        Args:
            structured_query: StructuredQuery node to be visited.

        Returns:
            Translated query and query kwargs.
        """
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"expr": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
