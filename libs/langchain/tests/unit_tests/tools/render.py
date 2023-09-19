from langchain.tools.base import tool
from langchain.tools.render import (
    render_text_description,
    render_text_description_and_args,
)


@tool
def search(query: str) -> str:
    """Lookup things online."""
    pass


@tool
def calculator(expression: str) -> str:
    """Do math."""
    pass


def test_render_text_description() -> None:
    tool_string = render_text_description([search, calculator])
    expected_string = """search: search(query: str) -> str - Lookup things online.
calculator: calculator(expression: str) -> str - Do math."""
    assert tool_string == expected_string


def test_render_text_description_and_args() -> None:
    tool_string = render_text_description_and_args([search, calculator])
    expected_string = """search: search(query: str) -> str - Lookup things online., args: {'query': {'title': 'Query', 'type': 'string'}}
calculator: calculator(expression: str) -> str - Do math., args: {'expression': {'title': 'Expression', 'type': 'string'}}"""
    assert tool_string == expected_string
