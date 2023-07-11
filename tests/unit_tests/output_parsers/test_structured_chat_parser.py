import pytest

from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


@pytest.fixture
def parser():
    return StructuredChatOutputParser()


def test_parse_expected_format(parser) -> None:
    text = """
    Action:
    ```
    {
      "action": "calculator",
      "action_input": {"operator": "+", "x": 1, "y": 2}
    }
    ```
    """
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "calculator"
    assert result.tool_input == {"operator": "+", "x": 1, "y": 2}


def test_parse_variant_format(parser) -> None:
    # This format is sometimes produced by AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION when running chat-bison
    # This format is the same as the expected format, but lacks a newline and the triple backtick delimiters
    text = """
    Action: {
      "action": "calculator",
      "action_input": {"operator": "+", "x": 1, "y": 2}
    }
    """
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "calculator"
    assert result.tool_input == {"operator": "+", "x": 1, "y": 2}


def test_parse_final_answer_expected_format(parser) -> None:
    text = """
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": 3
    }
    ```
    """
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.return_values == {"output": 3}


def test_parse_final_answer_variant_format(parser) -> None:
    # This format is sometimes produced by AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION when running chat-bison
    # This format is the same as the expected format, but lacks a newline and the triple backtick delimiters
    text = """
    Action: {
      "action": "Final Answer",
      "action_input": 3
    }
    """
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.return_values == {"output": 3}


def test_parse_no_action(parser) -> None:
    text = "I am a bot, how can I help you?"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.return_values == {"output": text}


def test_parse_invalid_json(parser) -> None:
    text = """
    Action:
    ```
    {
      "action": "calculator",
      "action_input": {"operator": "+", "x": 1, "y": 2}
    ```
    """  # Missing closing curly bracket
    with pytest.raises(OutputParserException):
        parser.parse(text)
