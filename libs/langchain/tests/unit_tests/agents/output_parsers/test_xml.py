import pytest
from langchain_core.agents import AgentAction, AgentFinish

from langchain.agents.format_scratchpad.xml import format_xml
from langchain.agents.output_parsers.xml import XMLAgentOutputParser


def test_parser_tool_usage() -> None:
    """Tests parsing a standard tool invocation."""
    parser = XMLAgentOutputParser(unescape_xml=False)
    _input = "<tool>search</tool><tool_input>foo</tool_input>"
    output = parser.invoke(_input)
    expected = AgentAction(tool="search", tool_input="foo", log=_input)
    assert output == expected


def test_parser_final_answer() -> None:
    """Tests parsing a standard final answer."""
    parser = XMLAgentOutputParser(unescape_xml=False)
    _input = "<final_answer>bar</final_answer>"
    output = parser.invoke(_input)
    expected = AgentFinish(return_values={"output": "bar"}, log=_input)
    assert output == expected


def test_parser_with_escaped_content() -> None:
    """Tests that the parser correctly unescapes standard XML entities."""
    parser = XMLAgentOutputParser(unescape_xml=True)
    _input = (
        "<tool>&lt;tool&gt; &amp; &apos;some_tool&apos;</tool>"
        "<tool_input>&lt;query&gt; with &quot;quotes&quot;</tool_input>"
    )

    output = parser.invoke(_input)

    expected = AgentAction(
        tool="<tool> & 'some_tool'",
        tool_input='<query> with "quotes"',
        log=_input,
    )
    assert output == expected


def test_parser_final_answer_escaped() -> None:
    """Tests parsing a final answer with escaped content."""
    parser = XMLAgentOutputParser(unescape_xml=True)
    _input = "<final_answer>The answer is &gt; 42.</final_answer>"

    output = parser.invoke(_input)

    expected = AgentFinish(return_values={"output": "The answer is > 42."}, log=_input)
    assert output == expected


def test_parser_error_on_multiple_tool_tags() -> None:
    """Tests that the parser raises an error for multiple tool tags."""
    parser = XMLAgentOutputParser()
    _input = "<tool>tool1</tool><tool>tool2</tool><tool_input>input</tool_input>"

    with pytest.raises(ValueError, match="Found 2 <tool> blocks"):
        parser.invoke(_input)


def test_parser_error_on_missing_required_tag() -> None:
    """Tests that the parser raises an error if a required tag is missing."""
    parser = XMLAgentOutputParser()
    _input = "<tool_input>some_input</tool_input>"

    with pytest.raises(ValueError, match="Could not parse LLM output"):
        parser.invoke(_input)


def test_integration_format_and_parse() -> None:
    """An integration test to ensure formatting and parsing work together."""
    parser = XMLAgentOutputParser()
    agent_action = AgentAction(tool="<special-tool>", tool_input="input with &", log="")
    intermediate_steps = [(agent_action, "observation with >")]

    # 1. Format the data, escaping the XML
    formatted_xml = format_xml(intermediate_steps, escape_xml=True)

    # Extract the tool call part for parsing
    tool_part = formatted_xml.split("<observation>")[0]

    # 2. Parse the formatted data
    output = parser.invoke(tool_part)

    # 3. Assert that the original data is recovered
    expected_action = AgentAction(
        tool="<special-tool>", tool_input="input with &", log=tool_part
    )
    assert output == expected_action
