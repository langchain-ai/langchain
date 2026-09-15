from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from langchain.agents import create_agent
from langchain.tools import tool
from tests.unit_tests.agents.model import FakeToolCallingModel


class InvalidToolCallingModel(FakeToolCallingModel):
    invalid_tool_call_id: str | None = "call_1"
    received_messages: list[BaseMessage] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = (stop, run_manager, kwargs)
        self.received_messages = messages
        message = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "get_weather",
                    "args": '{"city":',
                    "id": self.invalid_tool_call_id,
                    "error": "Invalid JSON",
                }
            ],
        )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return city


def test_create_agent_answers_invalid_tool_calls() -> None:
    model = InvalidToolCallingModel()
    agent = create_agent(model, [get_weather])

    result = agent.invoke({"messages": [HumanMessage("Weather?")]})

    assert model.index == 1
    assert len(result["messages"]) == 3
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.name == "get_weather"
    assert tool_message.status == "error"
    assert "malformed or truncated" in tool_message.text


def test_create_agent_answers_historical_invalid_tool_calls() -> None:
    model = InvalidToolCallingModel(invalid_tool_call_id=None)
    agent = create_agent(model, [get_weather])
    invalid_message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "get_weather",
                "args": '{"city":',
                "id": "historical_call",
                "error": "Invalid JSON",
            }
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("Weather?"), invalid_message]})

    historical_result = model.received_messages[-1]
    assert isinstance(historical_result, ToolMessage)
    assert historical_result.tool_call_id == "historical_call"
    assert (
        sum(
            isinstance(message, ToolMessage) and message.tool_call_id == "historical_call"
            for message in result["messages"]
        )
        == 1
    )


def test_create_agent_does_not_duplicate_historical_tool_messages() -> None:
    model = InvalidToolCallingModel(invalid_tool_call_id=None)
    agent = create_agent(model, [get_weather])
    invalid_message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "get_weather",
                "args": '{"city":',
                "id": "answered_call",
                "error": "Invalid JSON",
            }
        ],
    )
    existing_result = ToolMessage(
        content="Already answered",
        tool_call_id="answered_call",
        status="error",
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Weather?"), invalid_message, existing_result]}
    )

    assert model.received_messages[-1] is existing_result
    assert (
        sum(
            isinstance(message, ToolMessage) and message.tool_call_id == "answered_call"
            for message in result["messages"]
        )
        == 1
    )


def test_create_agent_ignores_invalid_tool_calls_without_ids() -> None:
    model = InvalidToolCallingModel(invalid_tool_call_id=None)
    agent = create_agent(model, [get_weather])

    result = agent.invoke({"messages": [HumanMessage("Weather?")]})

    assert model.index == 1
    assert len(result["messages"]) == 2
