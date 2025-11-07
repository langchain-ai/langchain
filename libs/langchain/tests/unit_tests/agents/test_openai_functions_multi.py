import json

import pytest
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, SystemMessage

from langchain_classic.agents.openai_functions_multi_agent.base import (
    _FunctionsAgentAction,
    _parse_ai_message,
)


# Test: _parse_ai_message() function.
class TestParseAIMessage:
    # Test: Pass Non-AIMessage.
    def test_not_an_ai(self) -> None:
        err = f"Expected an AI message got {SystemMessage!s}"
        with pytest.raises(TypeError, match=err):
            _parse_ai_message(SystemMessage(content="x"))

    # Test: Model response (not a function call).
    def test_model_response(self) -> None:
        msg = AIMessage(content="Model response.")
        result = _parse_ai_message(msg)

        assert isinstance(result, AgentFinish)
        assert result.return_values == {"output": "Model response."}
        assert result.log == "Model response."

    # Test: Model response with a function call.
    def test_func_call(self) -> None:
        act = json.dumps([{"action_name": "foo", "action": {"param": 42}}])

        msg = AIMessage(
            content="LLM thoughts.",
            additional_kwargs={
                "function_call": {"name": "foo", "arguments": f'{{"actions": {act}}}'},
            },
        )
        result = _parse_ai_message(msg)

        assert isinstance(result, list)
        assert len(result) == 1

        action = result[0]
        assert isinstance(action, _FunctionsAgentAction)
        assert action.tool == "foo"
        assert action.tool_input == {"param": 42}
        assert action.log == (
            "\nInvoking: `foo` with `{'param': 42}`\nresponded: LLM thoughts.\n\n"
        )
        assert action.message_log == [msg]

    # Test: Model response with a function call (old style tools).
    def test_func_call_oldstyle(self) -> None:
        act = json.dumps([{"action_name": "foo", "action": {"__arg1": "42"}}])

        msg = AIMessage(
            content="LLM thoughts.",
            additional_kwargs={
                "function_call": {"name": "foo", "arguments": f'{{"actions": {act}}}'},
            },
        )
        result = _parse_ai_message(msg)

        assert isinstance(result, list)
        assert len(result) == 1

        action = result[0]
        assert isinstance(action, _FunctionsAgentAction)
        assert action.tool == "foo"
        assert action.tool_input == "42"
        assert action.log == (
            "\nInvoking: `foo` with `42`\nresponded: LLM thoughts.\n\n"
        )
        assert action.message_log == [msg]

    # Test: Invalid function call args.
    def test_func_call_invalid(self) -> None:
        msg = AIMessage(
            content="LLM thoughts.",
            additional_kwargs={"function_call": {"name": "foo", "arguments": "{42]"}},
        )

        err = (
            "Could not parse tool input: {'name': 'foo', 'arguments': '{42]'} "
            "because the `arguments` is not valid JSON."
        )
        with pytest.raises(OutputParserException, match=err):
            _parse_ai_message(msg)
