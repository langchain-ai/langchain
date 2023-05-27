"""Test tool utils."""
import unittest
from typing import Any, Type
from unittest.mock import MagicMock, Mock

import pytest

from langchain.agents import load_tools
from langchain.agents.agent import Agent
from langchain.agents.chat.base import ChatAgent
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.react.base import ReActDocstoreAgent, ReActTextWorldAgent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent
from langchain.agents.tools import Tool, tool
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.parametrize(
    "agent_cls",
    [
        ZeroShotAgent,
        ChatAgent,
        ConversationalChatAgent,
        ConversationalAgent,
        ReActDocstoreAgent,
        ReActTextWorldAgent,
        SelfAskWithSearchAgent,
    ],
)
def test_single_input_agent_raises_error_on_structured_tool(
    agent_cls: Type[Agent],
) -> None:
    """Test that older agents raise errors on older tools."""

    @tool
    def the_tool(foo: str, bar: str) -> str:
        """Return the concat of foo and bar."""
        return foo + bar

    with pytest.raises(
        ValueError,
        match=f"{agent_cls.__name__} does not support"  # type: ignore
        f" multi-input tool {the_tool.name}.",
    ):
        agent_cls.from_llm_and_tools(MagicMock(), [the_tool])  # type: ignore


def test_tool_no_args_specified_assumes_str() -> None:
    """Older tools could assume *args and **kwargs were passed in."""

    def ambiguous_function(*args: Any, **kwargs: Any) -> str:
        """An ambiguously defined function."""
        return args[0]

    some_tool = Tool(
        name="chain_run",
        description="Run the chain",
        func=ambiguous_function,
    )
    expected_args = {"tool_input": {"type": "string"}}
    assert some_tool.args == expected_args
    assert some_tool.run("foobar") == "foobar"
    assert some_tool.run({"tool_input": "foobar"}) == "foobar"
    with pytest.raises(ValueError, match="Too many arguments to single-input tool"):
        some_tool.run({"tool_input": "foobar", "other_input": "bar"})


def test_load_tools_with_callback_manager_raises_deprecation_warning() -> None:
    """Test load_tools raises a deprecation for old callback manager kwarg."""
    callback_manager = MagicMock()
    with pytest.warns(DeprecationWarning, match="callback_manager is deprecated"):
        tools = load_tools(["requests_get"], callback_manager=callback_manager)
    assert len(tools) == 1
    assert tools[0].callbacks == callback_manager


def test_load_tools_with_callbacks_is_called() -> None:
    """Test callbacks are called when provided to load_tools fn."""
    callbacks = [FakeCallbackHandler()]
    tools = load_tools(["requests_get"], callbacks=callbacks)  # type: ignore
    assert len(tools) == 1
    # Patch the requests.get() method to return a mock response
    with unittest.mock.patch(
        "langchain.requests.TextRequestsWrapper.get",
        return_value=Mock(text="Hello world!"),
    ):
        result = tools[0].run("https://www.google.com")
        assert result.text == "Hello world!"
    assert callbacks[0].tool_starts == 1
    assert callbacks[0].tool_ends == 1
