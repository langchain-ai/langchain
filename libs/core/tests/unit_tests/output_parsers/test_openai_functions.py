import json
from typing import Any

import pytest
from pydantic import BaseModel

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.outputs import ChatGeneration


def test_json_output_function_parser() -> None:
    """Test the JSON output function parser is configured with robust defaults."""
    message = AIMessage(
        content="This is a test message",
        additional_kwargs={
            "function_call": {
                "name": "function_name",
                "arguments": '{"arg1": "code\ncode"}',
            }
        },
    )
    chat_generation = ChatGeneration(message=message)

    # Full output
    # Test that the parsers defaults are configured to parse in non-strict mode
    parser = JsonOutputFunctionsParser(args_only=False)
    result = parser.parse_result([chat_generation])
    assert result == {"arguments": {"arg1": "code\ncode"}, "name": "function_name"}

    # Args only
    parser = JsonOutputFunctionsParser(args_only=True)
    result = parser.parse_result([chat_generation])
    assert result == {"arg1": "code\ncode"}

    # Verify that the original message is not modified
    assert message.additional_kwargs == {
        "function_call": {
            "name": "function_name",
            "arguments": '{"arg1": "code\ncode"}',
        }
    }


@pytest.mark.parametrize(
    "config",
    [
        {
            "args_only": False,
            "strict": False,
            "args": '{"arg1": "value1"}',
            "result": {"arguments": {"arg1": "value1"}, "name": "function_name"},
            "exception": None,
        },
        {
            "args_only": True,
            "strict": False,
            "args": '{"arg1": "value1"}',
            "result": {"arg1": "value1"},
            "exception": None,
        },
        {
            "args_only": True,
            "strict": False,
            "args": '{"code": "print(2+\n2)"}',
            "result": {"code": "print(2+\n2)"},
            "exception": None,
        },
        {
            "args_only": True,
            "strict": False,
            "args": '{"code": "你好)"}',
            "result": {"code": "你好)"},
            "exception": None,
        },
        {
            "args_only": True,
            "strict": True,
            "args": '{"code": "print(2+\n2)"}',
            "exception": OutputParserException,
        },
    ],
)
def test_json_output_function_parser_strictness(config: dict[str, Any]) -> None:
    """Test parsing with JSON strictness on and off."""
    args = config["args"]

    message = AIMessage(
        content="This is a test message",
        additional_kwargs={
            "function_call": {"name": "function_name", "arguments": args}
        },
    )
    chat_generation = ChatGeneration(message=message)

    # Full output
    parser = JsonOutputFunctionsParser(
        strict=config["strict"], args_only=config["args_only"]
    )
    if config["exception"] is not None:
        with pytest.raises(config["exception"]):
            parser.parse_result([chat_generation])
    else:
        assert parser.parse_result([chat_generation]) == config["result"]


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


def test_pydantic_output_functions_parser() -> None:
    """Test pydantic output functions parser."""
    message = AIMessage(
        content="This is a test message",
        additional_kwargs={
            "function_call": {
                "name": "function_name",
                "arguments": json.dumps({"name": "value", "age": 10}),
            }
        },
    )
    chat_generation = ChatGeneration(message=message)

    class Model(BaseModel):
        """Test model."""

        name: str
        age: int

    # Full output
    parser = PydanticOutputFunctionsParser(pydantic_schema=Model)
    result = parser.parse_result([chat_generation])
    assert result == Model(name="value", age=10)


def test_pydantic_output_functions_parser_multiple_schemas() -> None:
    """Test that the parser works if providing multiple pydantic schemas."""
    message = AIMessage(
        content="This is a test message",
        additional_kwargs={
            "function_call": {
                "name": "cookie",
                "arguments": json.dumps({"name": "value", "age": 10}),
            }
        },
    )
    chat_generation = ChatGeneration(message=message)

    class Cookie(BaseModel):
        """Test model."""

        name: str
        age: int

    class Dog(BaseModel):
        """Test model."""

        species: str

    # Full output
    parser = PydanticOutputFunctionsParser(
        pydantic_schema={"cookie": Cookie, "dog": Dog}
    )
    result = parser.parse_result([chat_generation])
    assert result == Cookie(name="value", age=10)
