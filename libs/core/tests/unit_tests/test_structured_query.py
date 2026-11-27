"""Tests for the structured query expression visitor dispatch."""

from typing import Any

from langchain_core.structured_query import Comparison, Visitor


class APIRequest(Comparison):
    """Expression with a consecutive-capital class name."""


class RecordingVisitor(Visitor):
    def visit_operation(self, operation: Any) -> Any:
        return operation

    def visit_comparison(self, comparison: Any) -> Any:
        return comparison

    def visit_structured_query(self, structured_query: Any) -> Any:
        return structured_query

    def visit_api_request(self, expr: APIRequest) -> APIRequest:
        return expr


def test_accept_dispatches_acronym_class_name_to_snake_case_method() -> None:
    expr = APIRequest(comparator="eq", attribute="status", value=200)

    assert expr.accept(RecordingVisitor()) is expr
