from langchain.agents.conversational_chat.output_parser import \
    ConvoOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


def test_parse() -> None:
    parser = ConvoOutputParser()

    # Test valid JSON input
    text = '```json\n{\n"action": "Current Search",\n"action_input": "Thai food dinner recipes"\n}\n```'
    expected_action="Current Search"
    expected_result = "Thai food dinner recipes"
    result = parser.parse(text)
    assert result == AgentAction(expected_action, expected_result, text)

    # Test valid input without JSON
    text = '```\n{\n"action": "Current Search",\n"action_input": "Thai food dinner recipes"\n}\n```'
    expected_action="Current Search"
    expected_result = "Thai food dinner recipes"
    result = parser.parse(text)
    assert result == AgentAction(expected_action, expected_result, text)

    # Test final answer exhuasted tokens JSON
    text = '```json\n{\n"action": "Final Answer",\n"action_input": "Thai food dinner '
    expected_result = "Thai food dinner"
    result = parser.parse(text)
    assert result == AgentFinish({"output": expected_result}, text)

    # Test no JSON
    text = 'Thai food dinner recipes'
    expected_result = "Thai food dinner recipes"
    result = parser.parse(text)
    assert result == AgentFinish({"output": expected_result}, text)

    # Test invalid JSON input
    text = '```\n{\n"action": "Current Search",\n"action_input": "Thai food "dinner" recipes"\n}\n```'
    try:
        parser.parse(text)
    except OutputParserException:
        pass  # Test passes if OutputParserException is raised
    else:
        assert False, f"Expected OutputParserException, but got {parser.parse(text)}"