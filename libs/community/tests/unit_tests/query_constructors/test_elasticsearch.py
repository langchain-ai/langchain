from typing import Dict, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator

DEFAULT_TRANSLATOR = ElasticsearchTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.EQ, attribute="foo", value="1")
    expected = {"term": {"metadata.foo.keyword": "1"}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_range_gt() -> None:
    comp = Comparison(comparator=Comparator.GT, attribute="foo", value=1)
    expected = {"range": {"metadata.foo": {"gt": 1}}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_range_gte() -> None:
    comp = Comparison(comparator=Comparator.GTE, attribute="foo", value=1)
    expected = {"range": {"metadata.foo": {"gte": 1}}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_range_lt() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = {"range": {"metadata.foo": {"lt": 1}}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_range_lte() -> None:
    comp = Comparison(comparator=Comparator.LTE, attribute="foo", value=1)
    expected = {"range": {"metadata.foo": {"lte": 1}}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_range_match() -> None:
    comp = Comparison(comparator=Comparator.CONTAIN, attribute="foo", value="1")
    expected = {"match": {"metadata.foo": {"query": "1"}}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_range_like() -> None:
    comp = Comparison(comparator=Comparator.LIKE, attribute="foo", value="bar")
    expected = {"match": {"metadata.foo": {"query": "bar", "fuzziness": "AUTO"}}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = {
        "bool": {
            "must": [
                {"term": {"metadata.foo": 2}},
                {"term": {"metadata.bar.keyword": "baz"}},
            ]
        }
    }
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_operation_or() -> None:
    op = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = {
        "bool": {
            "should": [
                {"term": {"metadata.foo": 2}},
                {"term": {"metadata.bar.keyword": "baz"}},
            ]
        }
    }
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_operation_not() -> None:
    op = Operation(
        operator=Operator.NOT,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = {
        "bool": {
            "must_not": [
                {"term": {"metadata.foo": 2}},
                {"term": {"metadata.bar.keyword": "baz"}},
            ]
        }
    }
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_structured_query() -> None:
    query = "What is the capital of France?"

    structured_query = StructuredQuery(query=query, filter=None, limit=None)
    expected: Tuple[str, Dict] = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_filter() -> None:
    query = "What is the capital of France?"
    comp = Comparison(comparator=Comparator.EQ, attribute="foo", value="1")
    structured_query = StructuredQuery(query=query, filter=comp, limit=None)
    expected = (
        query,
        {"filter": [{"term": {"metadata.foo.keyword": "1"}}]},
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_filter_and() -> None:
    query = "What is the capital of France?"
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    structured_query = StructuredQuery(query=query, filter=op, limit=None)
    expected = (
        query,
        {
            "filter": [
                {
                    "bool": {
                        "must": [
                            {"term": {"metadata.foo": 2}},
                            {"term": {"metadata.bar.keyword": "baz"}},
                        ]
                    }
                }
            ]
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_complex() -> None:
    query = "What is the capital of France?"
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Operation(
                operator=Operator.OR,
                arguments=[
                    Comparison(comparator=Comparator.LT, attribute="bar", value=1),
                    Comparison(comparator=Comparator.LIKE, attribute="bar", value="10"),
                ],
            ),
        ],
    )
    structured_query = StructuredQuery(query=query, filter=op, limit=None)
    expected = (
        query,
        {
            "filter": [
                {
                    "bool": {
                        "must": [
                            {"term": {"metadata.foo": 2}},
                            {
                                "bool": {
                                    "should": [
                                        {"range": {"metadata.bar": {"lt": 1}}},
                                        {
                                            "match": {
                                                "metadata.bar": {
                                                    "query": "10",
                                                    "fuzziness": "AUTO",
                                                }
                                            }
                                        },
                                    ]
                                }
                            },
                        ]
                    }
                }
            ]
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_with_date_range() -> None:
    query = "Who was the president of France in 1995?"
    operation = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value="20"),
            Operation(
                operator=Operator.AND,
                arguments=[
                    Comparison(
                        comparator=Comparator.GTE,
                        attribute="timestamp",
                        value={"date": "1995-01-01", "type": "date"},
                    ),
                    Comparison(
                        comparator=Comparator.LT,
                        attribute="timestamp",
                        value={"date": "1996-01-01", "type": "date"},
                    ),
                ],
            ),
        ],
    )
    structured_query = StructuredQuery(query=query, filter=operation, limit=None)
    expected = (
        query,
        {
            "filter": [
                {
                    "bool": {
                        "must": [
                            {"term": {"metadata.foo.keyword": "20"}},
                            {
                                "bool": {
                                    "must": [
                                        {
                                            "range": {
                                                "metadata.timestamp": {
                                                    "gte": "1995-01-01"
                                                }
                                            }
                                        },
                                        {
                                            "range": {
                                                "metadata.timestamp": {
                                                    "lt": "1996-01-01"
                                                }
                                            }
                                        },
                                    ]
                                }
                            },
                        ]
                    }
                }
            ]
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_with_date() -> None:
    query = "Who was the president of France on 1st of January 1995?"
    operation = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value="20"),
            Comparison(
                comparator=Comparator.EQ,
                attribute="timestamp",
                value={"date": "1995-01-01", "type": "date"},
            ),
        ],
    )
    structured_query = StructuredQuery(query=query, filter=operation, limit=None)
    expected = (
        query,
        {
            "filter": [
                {
                    "bool": {
                        "must": [
                            {"term": {"metadata.foo.keyword": "20"}},
                            {"term": {"metadata.timestamp": "1995-01-01"}},
                        ]
                    }
                }
            ]
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
