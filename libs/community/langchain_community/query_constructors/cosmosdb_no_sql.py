from typing import Any, Dict, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

SQL_COMPARATOR = {
    Comparator.EQ: "=",
    Comparator.NE: "!=",
    Comparator.GT: ">",
    Comparator.GTE: ">=",
    Comparator.LT: "<",
    Comparator.LTE: "<=",
    Comparator.LIKE: "LIKE",
    Comparator.IN: "IN",
    Comparator.NIN: "NOT IN",
}

SQL_OPERATOR = {
    Operator.AND: "AND",
    Operator.OR: "OR",
    Operator.NOT: "NOT",
}


class AzureCosmosDbNoSQLTranslator(Visitor):
    """
    A visitor subclass that converts a StructuredQuery into an CosmosDB NO SQL query.
    """

    def __init__(self, table_name: str = "c") -> None:
        self.table_name = table_name

    def visit_comparison(self, comparison: Comparison) -> str:
        """
        Visit a comparison operation and convert it into an SQL condition.
        """
        operator = SQL_COMPARATOR.get(comparison.comparator)
        value = comparison.value
        field = f"{self.table_name}.{comparison.attribute}"

        if operator is None:
            raise ValueError(f"Unsupported operator: {comparison.comparator}")

        # Correct value formatting
        if isinstance(value, str):
            value = f"'{value}'"
        elif isinstance(value, (list, tuple)):  # Handle IN clause
            if comparison.comparator not in [Comparator.IN, Comparator.NIN]:
                raise ValueError(
                    f"Invalid comparator for list value: {comparison.comparator}"
                )
            value = (
                "("
                + ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
                + ")"
            )

        return f"{field} {operator} {value}"

    def visit_operation(self, operation: Operation) -> str:
        """
        Visit logical operations and convert them into SQL expressions.
        Uses parentheses to ensure correct precedence.
        """
        operator = SQL_OPERATOR.get(operation.operator)
        if operator is None:
            raise ValueError(f"Unsupported operator: {operation.operator}")

        expressions = [arg.accept(self) for arg in operation.arguments]

        if operation.operator == Operator.NOT:
            return f"NOT ({expressions[0]})"

        return f"({f' {operator} '.join(expressions)})"

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, Dict[str, Any]]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"where": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
