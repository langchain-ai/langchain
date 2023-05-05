"""Test tool utils."""
from typing import Any, Type
from unittest.mock import MagicMock

import pytest

from langchain.agents.agent import Agent
from langchain.agents.chat.base import ChatAgent
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.react.base import ReActDocstoreAgent, ReActTextWorldAgent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent
from langchain.agents.tools import Tool, tool


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
