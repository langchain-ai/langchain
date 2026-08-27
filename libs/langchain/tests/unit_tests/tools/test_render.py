import pytest
from langchain_core.tools import BaseTool, tool

from langchain_classic.tools.render import (
    render_text_description,
    render_text_description_and_args,
)


@tool
def search(query: str) -> str:  # noqa: ARG001
    """Lookup things online."""
    return "foo"


@tool
def calculator(expression: str) -> str:  # noqa: ARG001
    """Do math."""
    return "bar"


@pytest.fixture
def tools() -> list[BaseTool]:
    return [search, calculator]


def test_render_text_description(tools: list[BaseTool]) -> None:
    tool_string = render_text_description(tools)
    expected_string = """search(query: str) -> str - Lookup things online.
calculator(expression: str) -> str - Do math."""
    assert tool_string == expected_string


def test_render_text_description_and_args(tools: list[BaseTool]) -> None:
    tool_string = render_text_description_and_args(tools)
    expected_string = """search(query: str) -> str - Lookup things online., \
args: {'query': {'title': 'Query', 'type': 'string'}}
calculator(expression: str) -> str - Do math., \
args: {'expression': {'title': 'Expression', 'type': 'string'}}"""
    assert tool_string == expected_string
