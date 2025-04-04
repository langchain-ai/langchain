from typing import Any, Dict, List, Union

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    FilterDirective,
    Operation,
    Operator,
)

# Mapping of MongoDB comparison operators to LangChain comparators
MQL_TO_LC_COMPARATOR = {
    "$eq": Comparator.EQ,
    "$ne": Comparator.NE,
    "$gt": Comparator.GT,
    "$gte": Comparator.GTE,
    "$lt": Comparator.LT,
    "$lte": Comparator.LTE,
    "$in": Comparator.IN,
    "$nin": Comparator.NIN,
}

MQL_LOGICAL_OPERATORS = {
    "$and": Operator.AND,
    "$or": Operator.OR,
    "$not": Operator.NOT,
}


def _parse_condition(
    field: str, condition: Union[Dict[str, Any], List[Any], Any]
) -> FilterDirective:
    """
    Convert a mql condition dict to a FilterDirective.
    Handles implicit equality, implicit $in, and explicit operators.

    Args:
        field (str): The field name in the mql condition.
        condition (Union[Dict[str, Any], List[Any], Any]): The condition to be parsed.

    Returns:
        FilterDirective: The parsed condition.
    """
    # Handle implicit IN operator
    if isinstance(condition, list):
        # validate elements are simple scalars
        if not all(isinstance(x, (int, float, str, bool)) for x in condition):
            raise ValueError(
                f"List condition in {field} must contain only scalar values."
            )
        return Comparison(comparator=Comparator.IN, attribute=field, value=condition)

    if isinstance(condition, dict) and len(condition) == 0:
        raise ValueError(f"Implicit $exists not supported, in {field}")

    # Handle implicit equality
    if not isinstance(condition, dict):
        return Comparison(comparator=Comparator.EQ, attribute=field, value=condition)

    # handle explicit operators (and implicit $and)
    expressions: List[FilterDirective] = []
    for op, value in condition.items():
        if op not in MQL_TO_LC_COMPARATOR.keys():
            raise ValueError(f"Unsupported mql operator: {op}")

        # validate value types
        # $in $nin must be a list
        if op in ["$in", "$nin"] and not isinstance(value, list):
            raise ValueError(f"{op} operator must be applied to a list.")
        # rest must be scalars
        if op not in ["$in", "$nin"] and not isinstance(value, (int, float, str, bool)):
            raise ValueError(f"{op} operator must be applied to a scalar value.")

        expressions.append(
            Comparison(
                comparator=MQL_TO_LC_COMPARATOR[op], attribute=field, value=value
            )
        )

    return (
        expressions[0]
        if len(expressions) == 1
        else Operation(operator=Operator.AND, arguments=expressions)
    )


def _parse_logical_operator(
    operator: str, conditions: List[FilterDirective]
) -> Operation:
    """
    Parse a mql logical operator into an Operation.

    Args:
        operator (str): The logical operator to be parsed.
        conditions (List[FilterDirective]): The conditions to be combined
            by the logical operator.

    Returns:
        Operation: The parsed logical operator.
    """
    if operator == "$not":
        if len(conditions) != 1:
            raise ValueError("$not operator must be applied to a single condition.")
        return Operation(operator=Operator.NOT, arguments=conditions)

    if operator == "$nor":
        return Operation(
            operator=Operator.NOT,
            arguments=[Operation(operator=Operator.OR, arguments=conditions)],
        )

    if operator not in MQL_LOGICAL_OPERATORS:
        raise ValueError(f"Unsupported mql logical operator: {operator}")

    return Operation(operator=MQL_LOGICAL_OPERATORS[operator], arguments=conditions)


def mql_to_filter(mql_filter: Dict[str, Any]) -> FilterDirective:
    """
    Convert a mql-like filter dict to a FilterDirective.

    Args:
        mql_filter (Dict[str, Any]): The mql filter to be converted.

    Returns:
        FilterDirective: The converted filter.
    """
    if not isinstance(mql_filter, dict):
        raise ValueError("MQL filter must be a dictionary.")

    expressions: list[FilterDirective] = []
    for field, expr in mql_filter.items():
        if field in MQL_LOGICAL_OPERATORS or field == "$nor":
            if not isinstance(expr, list) and field != "$not":
                raise ValueError(
                    f"Logical operator '{field}' must have a list of conditions."
                )

            operands: list[FilterDirective] = []

            for sub in expr if isinstance(expr, list) else [expr]:
                if not isinstance(sub, dict):
                    raise ValueError("Logical operator operands must be dictionaries.")
                operands.append(mql_to_filter(sub))
            expressions.append(_parse_logical_operator(field, operands))
        elif field.startswith("$"):
            if field in MQL_TO_LC_COMPARATOR.keys():
                raise ValueError(
                    f"Comparison operator '{field}' must be applied to a field."
                )

            raise ValueError(f"Unsupported MQL operator: {field}")
        else:
            expressions.append(_parse_condition(field, expr))

    return (
        expressions[0]
        if len(expressions) == 1
        else Operation(operator=Operator.AND, arguments=expressions)
    )
