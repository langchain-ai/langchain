"""Tests for structured_query module."""

import pytest

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    FilterDirective,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


def test_operator_enum():
    """Test Operator enum values."""
    assert Operator.AND == "and"
    assert Operator.OR == "or"
    assert Operator.NOT == "not"


def test_comparator_enum():
    """Test Comparator enum values."""
    assert Comparator.EQ == "eq"
    assert Comparator.NE == "ne"
    assert Comparator.GT == "gt"
    assert Comparator.GTE == "gte"
    assert Comparator.LT == "lt"
    assert Comparator.LTE == "lte"
    assert Comparator.CONTAIN == "contain"
    assert Comparator.LIKE == "like"
    assert Comparator.IN == "in"
    assert Comparator.NIN == "nin"


def test_comparison_creation():
    """Test Comparison creation."""
    comp = Comparison(
        comparator=Comparator.EQ,
        attribute="name",
        value="test",
    )
    assert comp.comparator == Comparator.EQ
    assert comp.attribute == "name"
    assert comp.value == "test"


def test_operation_creation():
    """Test Operation creation."""
    comp1 = Comparison(comparator=Comparator.EQ, attribute="a", value=1)
    comp2 = Comparison(comparator=Comparator.GT, attribute="b", value=2)
    op = Operation(operator=Operator.AND, arguments=[comp1, comp2])
    assert op.operator == Operator.AND
    assert len(op.arguments) == 2


def test_structured_query_creation():
    """Test StructuredQuery creation."""
    comp = Comparison(comparator=Comparator.EQ, attribute="x", value=10)
    sq = StructuredQuery(query="test", filter=comp, limit=5)
    assert sq.query == "test"
    assert sq.filter == comp
    assert sq.limit == 5


def test_visitor_validate_operator_error_message():
    """Test that operator validation error has correct message.

    This test verifies the fix for a bug where the error message
    incorrectly mentioned 'comparators' instead of 'operators'.
    """

    class TestVisitor(Visitor):
        allowed_operators = [Operator.AND]
        allowed_comparators = [Comparator.EQ]

        def visit_operation(self, operation):
            pass

        def visit_comparison(self, comparison):
            pass

        def visit_structured_query(self, structured_query):
            pass

    visitor = TestVisitor()

    # Should raise ValueError with correct error message
    with pytest.raises(ValueError) as exc_info:
        visitor._validate_func(Operator.OR)

    error_msg = str(exc_info.value)
    # Verify error message mentions 'operators' not 'comparators'
    assert "operators are" in error_msg
    assert "Allowed operators" in error_msg


def test_visitor_validate_comparator_error_message():
    """Test that comparator validation error has correct message."""

    class TestVisitor(Visitor):
        allowed_operators = [Operator.AND]
        allowed_comparators = [Comparator.EQ]

        def visit_operation(self, operation):
            pass

        def visit_comparison(self, comparison):
            pass

        def visit_structured_query(self, structured_query):
            pass

    visitor = TestVisitor()

    # Should raise ValueError with correct error message
    with pytest.raises(ValueError) as exc_info:
        visitor._validate_func(Comparator.GT)

    error_msg = str(exc_info.value)
    assert "comparators are" in error_msg
    assert "Allowed comparators" in error_msg


def test_visitor_validate_allowed():
    """Test that allowed operators/comparators pass validation."""

    class TestVisitor(Visitor):
        allowed_operators = [Operator.AND]
        allowed_comparators = [Comparator.EQ]

        def visit_operation(self, operation):
            pass

        def visit_comparison(self, comparison):
            pass

        def visit_structured_query(self, structured_query):
            pass

    visitor = TestVisitor()

    # Should not raise
    visitor._validate_func(Operator.AND)
    visitor._validate_func(Comparator.EQ)


def test_comparison_accept():
    """Test that Comparison can accept a visitor."""
    results = []

    class TestVisitor(Visitor):
        def visit_comparison(self, comparison):
            results.append(("comparison", comparison.attribute))
            return results[-1]

        def visit_operation(self, operation):
            results.append(("operation", operation.operator))
            return results[-1]

        def visit_structured_query(self, structured_query):
            results.append(("query", structured_query.query))
            return results[-1]

    visitor = TestVisitor()
    comp = Comparison(comparator=Comparator.EQ, attribute="test_attr", value="test")
    result = comp.accept(visitor)
    assert result == ("comparison", "test_attr")
