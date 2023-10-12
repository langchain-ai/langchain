from langchain.agents.output_parsers.react_json_single_input import (
    ReActJsonSingleInputOutputParser,
)
from langchain.schema.agent import AgentAction, AgentFinish


def test_action() -> None:
    """Test standard parsing of action/action input."""
    parser = ReActJsonSingleInputOutputParser()
    _input = """Thought: agent thought here
```
{
    "action": "search",
    "action_input": "what is the temperature in SF?"
}
```
"""
    output = parser.invoke(_input)
    expected_output = AgentAction(
        tool="search", tool_input="what is the temperature in SF?", log=_input
    )
    assert output == expected_output


def test_finish() -> None:
    """Test standard parsing of agent finish."""
    parser = ReActJsonSingleInputOutputParser()
    _input = """Thought: agent thought here
Final Answer: The temperature is 100"""
    output = parser.invoke(_input)
    expected_output = AgentFinish(
        return_values={"output": "The temperature is 100"}, log=_input
    )
    assert output == expected_output
