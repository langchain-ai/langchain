"""Test chain for interacting with SQL Database."""

import pytest

from langchain.chains.sql_database.base import SQLDatabaseChain


def test_normalize_sql_cmd_double_quotation_marks():
    sql_cmd = ' "SELECT * from fake "'
    result = SQLDatabaseChain.normalize_sql_cmd(sql_cmd)
    assert result == "SELECT * from fake"


def test_normalize_sql_cmd_single_quotation_marks():
    sql_cmd = " 'SELECT * from fake '"
    result = SQLDatabaseChain.normalize_sql_cmd(sql_cmd)
    assert result == "SELECT * from fake"


@pytest.mark.parametrize("sql_cmd", ["", '"', "'", "x", None])
def test_normalize_sql_cmd_no_change(sql_cmd):
    result = SQLDatabaseChain.normalize_sql_cmd(sql_cmd)
    assert result == sql_cmd  # no change


def test_normalize_sql_cmd_two_double_quotation_marks():
    sql_cmd = '""'
    result = SQLDatabaseChain.normalize_sql_cmd(sql_cmd)
    assert result == ""  # empty string
