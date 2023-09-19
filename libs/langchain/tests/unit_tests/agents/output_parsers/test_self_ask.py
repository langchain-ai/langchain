from langchain.agents.output_parsers.self_ask import SelfAskOutputParser
from langchain.schema.agent import AgentAction, AgentFinish


def test_follow_up() -> None:
    """Test follow up parsing."""
    parser = SelfAskOutputParser()
    _input = "Follow up: what is two + 2"
    output = parser.invoke(_input)
    expected_output = AgentAction(
        tool="Intermediate Answer", tool_input="what is two + 2", log=_input
    )
    assert output == expected_output
    # Test that also handles one word by default
    _input = "Followup: what is two + 2"
    output = parser.invoke(_input)
    expected_output = AgentAction(
        tool="Intermediate Answer", tool_input="what is two + 2", log=_input
    )
    assert output == expected_output


def test_follow_up_custom() -> None:
    """Test follow up parsing for custom followups."""
    parser = SelfAskOutputParser(followups=("Now:",))
    _input = "Now: what is two + 2"
    output = parser.invoke(_input)
    expected_output = AgentAction(
        tool="Intermediate Answer", tool_input="what is two + 2", log=_input
    )
    assert output == expected_output


def test_finish() -> None:
    """Test standard finish."""
    parser = SelfAskOutputParser()
    _input = "So the final answer is: 4"
    output = parser.invoke(_input)
    expected_output = AgentFinish(return_values={"output": "4"}, log=_input)
    assert output == expected_output


def test_finish_custom() -> None:
    """Test custom finish."""
    parser = SelfAskOutputParser(finish_string="Finally: ")
    _input = "Finally: 4"
    output = parser.invoke(_input)
    expected_output = AgentFinish(return_values={"output": "4"}, log=_input)
    assert output == expected_output
