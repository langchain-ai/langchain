import signal
import sys

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)


def test_action() -> None:
    """Test standard parsing of action/action input."""
    parser = ReActSingleInputOutputParser()
    _input = """Thought: agent thought here
Action: search
Action Input: what is the temperature in SF?"""
    output = parser.invoke(_input)
    expected_output = AgentAction(
        tool="search",
        tool_input="what is the temperature in SF?",
        log=_input,
    )
    assert output == expected_output


def test_finish() -> None:
    """Test standard parsing of agent finish."""
    parser = ReActSingleInputOutputParser()
    _input = """Thought: agent thought here
Final Answer: The temperature is 100"""
    output = parser.invoke(_input)
    expected_output = AgentFinish(
        return_values={"output": "The temperature is 100"},
        log=_input,
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


def _timeout_handler(_signum: int, _frame: object) -> None:
    msg = "ReDoS: regex took too long"
    raise TimeoutError(msg)


@pytest.mark.skipif(
    sys.platform == "win32", reason="SIGALRM is not available on Windows"
)
def test_react_single_input_no_redos() -> None:
    """Regression test for ReDoS caused by catastrophic backtracking."""
    parser = ReActSingleInputOutputParser()
    malicious = "Action: " + " \t" * 1000 + "Action "
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(2)
    try:
        try:
            parser.parse(malicious)
        except OutputParserException:
            pass
        except TimeoutError:
            pytest.fail(
                "ReDoS detected: ReActSingleInputOutputParser.parse() "
                "hung on crafted input"
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
