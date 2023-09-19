import json

from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.schema.messages import AIMessage


def test_tool_usage() -> None:
    parser = OpenAIFunctionsAgentOutputParser()
    function_call = json.dumps({"query": "hi"})
    _input = AIMessage(
        content="hi",
        additional_kwargs={
            "function_call": {"name": "search", "arguments": function_call}
        },
    )
    output = parser.invoke(_input)
    # Test this way, because we don't care about testing `log`
    assert output.tool == "search"
    assert output.tool_input == {"query": "hi"}
    assert output.message_log == [_input]


def test_finish() -> None:
    parser = OpenAIFunctionsAgentOutputParser()
    _input = AIMessage(content="hi")
    output = parser.invoke(_input)
    # Test this way, because we don't care about testing `log`
    assert output.return_values == {"output": "hi"}
