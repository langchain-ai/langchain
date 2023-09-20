import pytest

from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.schema import AgentFinish, OutputParserException
from langchain.schema.agent import AgentActionMessageLog
from langchain.schema.messages import AIMessage, SystemMessage


def test_not_an_ai() -> None:
    parser = OpenAIFunctionsAgentOutputParser()
    err = f"Expected an AI message got {str(SystemMessage)}"
    with pytest.raises(TypeError, match=err):
        parser.invoke(SystemMessage(content="x"))


# Test: Model response (not a function call).
def test_model_response() -> None:
    parser = OpenAIFunctionsAgentOutputParser()
    msg = AIMessage(content="Model response.")
    result = parser.invoke(msg)

    assert isinstance(result, AgentFinish)
    assert result.return_values == {"output": "Model response."}
    assert result.log == "Model response."


# Test: Model response with a function call.
def test_func_call() -> None:
    parser = OpenAIFunctionsAgentOutputParser()
    msg = AIMessage(
        content="LLM thoughts.",
        additional_kwargs={
            "function_call": {"name": "foo", "arguments": '{"param": 42}'}
        },
    )
    result = parser.invoke(msg)

    assert isinstance(result, AgentActionMessageLog)
    assert result.tool == "foo"
    assert result.tool_input == {"param": 42}
    assert result.log == (
        "\nInvoking: `foo` with `{'param': 42}`\nresponded: LLM thoughts.\n\n"
    )
    assert result.message_log == [msg]


# Test: Model response with a function call (old style tools).
def test_func_call_oldstyle() -> None:
    parser = OpenAIFunctionsAgentOutputParser()
    msg = AIMessage(
        content="LLM thoughts.",
        additional_kwargs={
            "function_call": {"name": "foo", "arguments": '{"__arg1": "42"}'}
        },
    )
    result = parser.invoke(msg)

    assert isinstance(result, AgentActionMessageLog)
    assert result.tool == "foo"
    assert result.tool_input == "42"
    assert result.log == "\nInvoking: `foo` with `42`\nresponded: LLM thoughts.\n\n"
    assert result.message_log == [msg]


# Test: Invalid function call args.
def test_func_call_invalid() -> None:
    parser = OpenAIFunctionsAgentOutputParser()
    msg = AIMessage(
        content="LLM thoughts.",
        additional_kwargs={"function_call": {"name": "foo", "arguments": "{42]"}},
    )

    err = (
        "Could not parse tool input: {'name': 'foo', 'arguments': '{42]'} "
        "because the `arguments` is not valid JSON."
    )
    with pytest.raises(OutputParserException, match=err):
        parser.invoke(msg)
