import pytest

from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.output_parser import OutputParserException


def test_action() -> None:
    """Test standard parsing of action/action input."""
    parser = ReActSingleInputOutputParser()
    _input = """Thought: agent thought here
Action: search
Action Input: what is the temperature in SF?"""
    output = parser.invoke(_input)
    expected_output = AgentAction(
        tool="search", tool_input="what is the temperature in SF?", log=_input
    )
    assert output == expected_output


def test_finish() -> None:
    """Test standard parsing of agent finish."""
    parser = ReActSingleInputOutputParser()
    _input = """Thought: agent thought here
Final Answer: The temperature is 100"""
    output = parser.invoke(_input)
    expected_output = AgentFinish(
        return_values={"output": "The temperature is 100"}, log=_input
    )
    assert output == expected_output


def test_action_with_finish() -> None:
    """Test that if final thought is in action/action input, error is raised."""
    parser = ReActSingleInputOutputParser()
    _input = """Thought: agent thought here
Action: search Final Answer:
Action Input: what is the temperature in SF?"""
    with pytest.raises(OutputParserException):
        parser.invoke(_input)
