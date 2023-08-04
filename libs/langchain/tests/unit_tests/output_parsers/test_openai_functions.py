import json

import pytest

from langchain.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
)
from langchain.schema import BaseMessage, ChatGeneration, OutputParserException
from langchain.schema.messages import AIMessage, HumanMessage


@pytest.fixture
def ai_message() -> AIMessage:
    """Return a simple AIMessage."""
    content = "This is a test message"

    args = json.dumps(
        {
            "arg1": "value1",
        }
    )

    function_call = {"name": "function_name", "arguments": args}
    additional_kwargs = {"function_call": function_call}
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


def test_json_output_function_parser(ai_message: AIMessage) -> None:
    """Test that the JsonOutputFunctionsParser with full output."""
    chat_generation = ChatGeneration(message=ai_message)

    # Full output
    parser = JsonOutputFunctionsParser(args_only=False)
    result = parser.parse_result([chat_generation])
    assert result == {"arguments": {"arg1": "value1"}, "name": "function_name"}

    # Args only
    parser = JsonOutputFunctionsParser(args_only=True)
    result = parser.parse_result([chat_generation])
    assert result == {"arg1": "value1"}

    # Verify that the original message is not modified
    assert ai_message.additional_kwargs == {
        "function_call": {"name": "function_name", "arguments": '{"arg1": "value1"}'}
    }


@pytest.mark.parametrize(
    "bad_message",
    [
        # Human message has no function call
        HumanMessage(content="This is a test message"),
        # AIMessage has no function call information.
        AIMessage(content="This is a test message", additional_kwargs={}),
        # Bad function call information (arguments should be a string)
        AIMessage(
            content="This is a test message",
            additional_kwargs={
                "function_call": {"name": "function_name", "arguments": {}}
            },
        ),
        # Bad function call information (arguments should be proper json)
        AIMessage(
            content="This is a test message",
            additional_kwargs={
                "function_call": {"name": "function_name", "arguments": "noqweqwe"}
            },
        ),
    ],
)
def test_exceptions_raised_while_parsing(bad_message: BaseMessage) -> None:
    """Test exceptions raised correctly while using JSON parser."""
    chat_generation = ChatGeneration(message=bad_message)

    with pytest.raises(OutputParserException):
        JsonOutputFunctionsParser().parse_result([chat_generation])
