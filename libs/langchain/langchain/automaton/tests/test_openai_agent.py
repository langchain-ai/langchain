from __future__ import annotations

from typing import List, cast

import pytest

from langchain.automaton.openai_agent import OpenAIAgent
from langchain.automaton.tests.utils import (
    FakeChatModel,
    construct_func_invocation_message,
)
from langchain.automaton.typedefs import (
    FunctionCall,
    FunctionResult,
    MessageLog,
    AgentFinish,
)
from langchain.schema.messages import (
    AIMessage,
    SystemMessage,
)
from langchain.tools import tool, Tool
from langchain.tools.base import BaseTool


@pytest.fixture()
def tools() -> List[BaseTool]:
    @tool
    def get_time() -> str:
        """Get time."""
        return "9 PM"

    @tool
    def get_location() -> str:
        """Get location."""
        return "the park"

    return cast(List[Tool], [get_time, get_location])


def test_openai_agent(tools: List[Tool]) -> None:
    get_time, get_location = tools
    llm = FakeChatModel(
        message_iter=iter(
            [
                construct_func_invocation_message(get_time, {}),
                AIMessage(
                    content="The time is 9 PM.",
                ),
            ]
        )
    )

    agent = OpenAIAgent(llm=llm, tools=tools, max_iterations=10)

    message_log = MessageLog(
        [
            SystemMessage(
                content="What time is it?",
            )
        ]
    )

    expected_messages = [
        SystemMessage(
            content="What time is it?",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "get_time",
                    "arguments": "{}",
                }
            },
        ),
        FunctionCall(
            name="get_time",
            arguments={},
        ),
        FunctionResult(
            name="get_time",
            result="9 PM",
            error=None,
        ),
        AIMessage(
            content="The time is 9 PM.",
        ),
        AgentFinish(
            AIMessage(
                content="The time is 9 PM.",
            ),
        ),
    ]

    agent.run(message_log)
    assert message_log.messages == expected_messages
