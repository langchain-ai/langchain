from langchain.agents.output_parsers.xml import XMLAgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish


def test_tool_usage() -> None:
    parser = XMLAgentOutputParser()
    # Test when final closing </tool_input> is included
    _input = """<tool>search</tool><tool_input>foo</tool_input>"""
    output = parser.invoke(_input)
    expected_output = AgentAction(tool="search", tool_input="foo", log=_input)
    assert output == expected_output
    # Test when final closing </tool_input> is NOT included
    # This happens when it's used as a stop token
    _input = """<tool>search</tool><tool_input>foo</tool_input>"""
    output = parser.invoke(_input)
    expected_output = AgentAction(tool="search", tool_input="foo", log=_input)
    assert output == expected_output


def test_finish() -> None:
    parser = XMLAgentOutputParser()
    # Test when final closing <final_answer> is included
    _input = """<final_answer>bar</final_answer>"""
    output = parser.invoke(_input)
    expected_output = AgentFinish(return_values={"output": "bar"}, log=_input)
    assert output == expected_output

    # Test when final closing <final_answer> is NOT included
    # This happens when it's used as a stop token
    _input = """<final_answer>bar</final_answer>"""
    output = parser.invoke(_input)
    expected_output = AgentFinish(return_values={"output": "bar"}, log=_input)
    assert output == expected_output
