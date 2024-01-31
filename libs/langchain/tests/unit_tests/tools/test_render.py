from typing import List

import pytest
from langchain_core.tools import BaseTool, tool

from langchain.tools.render import (
    render_text_description,
    render_text_description_and_args,
)


@tool
def search(query: str) -> str:
    """Lookup things online."""
    return "foo"


@tool
def calculator(expression: str) -> str:
    """Do math."""
    return "bar"


@pytest.fixture
def tools() -> List[BaseTool]:
    return [search, calculator]  # type: ignore


def test_render_text_description(tools: List[BaseTool]) -> None:
    tool_string = render_text_description(tools)
    expected_string = """search: search(query: str) -> str - Lookup things online.
calculator: calculator(expression: str) -> str - Do math."""
    assert tool_string == expected_string


def test_render_text_description_and_args(tools: List[BaseTool]) -> None:
    tool_string = render_text_description_and_args(tools)
    expected_string = """search: search(query: str) -> str - Lookup things online., \
args: {'query': {'title': 'Query', 'type': 'string'}}
calculator: calculator(expression: str) -> str - Do math., \
args: {'expression': {'title': 'Expression', 'type': 'string'}}"""
    assert tool_string == expected_string
