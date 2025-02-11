import pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_community.query_constructors.cosmosdb_no_sql import (
    AzureCosmosDbNoSQLTranslator,
)


def test_visit_structured_query_basic() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    structured_query = StructuredQuery(query="my search terms", limit=None, filter=None)
    query, filter = constructor.visit_structured_query(structured_query)
    assert query == "my search terms"
    assert filter == {}


def test_visit_structured_query_with_limit() -> None:
    constructor = AzureCosmosDbNoSQLTranslator(table_name="t")
    structured_query = StructuredQuery(query="my search terms", limit=10, filter=None)
    query, filter = constructor.visit_structured_query(structured_query)
    assert query == "my search terms"
    assert filter == {}


def test_visit_structured_query_with_filter() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    comparison = Comparison(attribute="age", comparator=Comparator.GT, value=30)
    structured_query = StructuredQuery(query="my search", limit=None, filter=comparison)
    query, filter = constructor.visit_structured_query(structured_query)
    assert query == "my search"
    assert filter == {"where": "c.age > 30"}


def test_visit_comparison_basic() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    comparison = Comparison(attribute="age", comparator=Comparator.GT, value=30)
    result = constructor.visit_comparison(comparison)
    assert result == "c.age > 30"


def test_visit_comparison_with_string() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    comparison = Comparison(attribute="name", comparator=Comparator.EQ, value="John")
    result = constructor.visit_comparison(comparison)
    assert result == "c.name = 'John'"


def test_visit_comparison_with_list() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    comparison = Comparison(
        attribute="age", comparator=Comparator.IN, value=[25, 30, 35]
    )
    result = constructor.visit_comparison(comparison)
    assert result == "c.age IN (25, 30, 35)"


def test_visit_comparison_unsupported_operator() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    comparison = Comparison(attribute="age", comparator=Comparator.CONTAIN, value=30)
    with pytest.raises(ValueError, match="Unsupported operator"):
        constructor.visit_comparison(comparison)


def test_visit_operation_basic() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    operation = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(attribute="age", comparator=Comparator.GT, value=30),
            Comparison(attribute="name", comparator=Comparator.EQ, value="John"),
        ],
    )
    result = constructor.visit_operation(operation)
    assert result == "(c.age > 30 AND c.name = 'John')"


def test_visit_operation_not() -> None:
    constructor = AzureCosmosDbNoSQLTranslator()
    operation = Operation(
        operator=Operator.NOT,
        arguments=[Comparison(attribute="age", comparator=Comparator.GT, value=30)],
    )
    result = constructor.visit_operation(operation)
    assert result == "NOT (c.age > 30)"
