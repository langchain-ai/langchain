"""Tests for SQL query validation in create_sql_query_chain."""

import pytest

from langchain_classic.chains.sql_database.query import validate_sql_query


def test_validate_sql_query_accepts_select() -> None:
    assert validate_sql_query("SELECT * FROM users") == "SELECT * FROM users"


def test_validate_sql_query_rejects_multiple_statements() -> None:
    with pytest.raises(ValueError, match="Multiple SQL statements"):
        validate_sql_query("SELECT 1; DROP TABLE users")


def test_validate_sql_query_rejects_non_select_by_default() -> None:
    with pytest.raises(ValueError, match="Only SELECT"):
        validate_sql_query("DELETE FROM users")


def test_validate_sql_query_rejects_forbidden_keywords() -> None:
    with pytest.raises(ValueError, match="DROP"):
        validate_sql_query("SELECT * FROM users WHERE id IN (DROP TABLE x)")
