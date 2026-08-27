from langchain_core.agents import AgentAction, AgentFinish

from langchain_classic.agents.output_parsers.xml import XMLAgentOutputParser


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


def test_malformed_xml_with_nested_tags() -> None:
    """Test handling of tool names with XML tags via format_xml minimal escaping."""
    from langchain_classic.agents.format_scratchpad.xml import format_xml

    # Create an AgentAction with XML tags in the tool name
    action = AgentAction(tool="search<tool>nested</tool>", tool_input="query", log="")

    # The format_xml function should escape the XML tags using custom delimiters
    formatted_xml = format_xml([(action, "observation")])

    # Extract just the tool part for parsing
    tool_part = formatted_xml.split("<observation>")[0]  # Remove observation part

    # Now test that the parser can handle the escaped XML
    parser = XMLAgentOutputParser(escape_format="minimal")
    output = parser.invoke(tool_part)

    # The parser should unescape and extract the original tool name
    expected_output = AgentAction(
        tool="search<tool>nested</tool>", tool_input="query", log=tool_part
    )
    assert output == expected_output


def test_no_escaping() -> None:
    """Test parser with escaping disabled."""
    parser = XMLAgentOutputParser(escape_format=None)

    # Test with regular tool name (no XML tags)
    _input = """<tool>search</tool><tool_input>foo</tool_input>"""
    output = parser.invoke(_input)
    expected_output = AgentAction(tool="search", tool_input="foo", log=_input)
    assert output == expected_output
