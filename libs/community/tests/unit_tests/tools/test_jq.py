"""Test functionality of JQ tools"""
import json

import pytest

from langchain_community.tools.jq.tool import JQ


@pytest.mark.requires("jq")
def test_jq_keys() -> None:
    """Test JQ can return keys of a dict at given path."""

    tool = JQ(query_expression="keys", return_type="first")
    out = tool(
        """
        {
               "foo": "bar", 
               "baz": {
                    "test": {
                        "foo": [1, 2, 3]
                    }
               }
        }"""
    )

    assert set(json.loads(out)) == {"foo", "baz"}


@pytest.mark.requires("jq")
def test_jq_values() -> None:
    """Test JQ can return keys of a dict at given path."""

    tool_input = """
        {
               "foo": "bar", 
               "baz": {
                    "test": {
                        "foo": [1, 2, 3]
                    }
               }
        }"""

    tool = JQ(query_expression=".foo", return_type="first")
    assert json.loads(tool(tool_input)) == "bar"

    tool = JQ(query_expression=".foo", return_type="all")
    assert json.loads(tool(tool_input)) == ["bar"]

    tool = JQ(query_expression=".foo", return_type="text")
    assert json.loads(tool(tool_input)) == "bar"

    tool = JQ(query_expression=".baz", return_type="first")
    assert json.loads(tool(tool_input)) == {"test": {"foo": [1, 2, 3]}}
    tool = JQ(query_expression=".baz", return_type="all")
    assert json.loads(tool(tool_input)) == [{"test": {"foo": [1, 2, 3]}}]

    tool = JQ(query_expression=".baz", return_type="text")
    assert json.loads(tool(tool_input)) == {"test": {"foo": [1, 2, 3]}}


@pytest.mark.requires("jq")
def test_jq_transform() -> None:
    """Test JQ can return keys of a dict at given path."""

    tool_input = """
        {
               "foo": "bar", 
               "baz": {
                    "test": {
                        "foo": [1, 2, 3]
                    }
               }
        }"""

    tool = JQ(query_expression=".baz.test.foo[]+1", return_type="first")
    assert json.loads(tool(tool_input)) == 2

    tool = JQ(query_expression=".baz.test.foo[]+1", return_type="all")
    assert json.loads(tool(tool_input)) == [2, 3, 4]

    tool = JQ(query_expression=".baz.test.foo[]+1", return_type="text")
    assert tool(tool_input) == "\n".join(str(e) for e in [2, 3, 4])
