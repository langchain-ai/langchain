from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


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
def get_weather(city: str = "Paris") -> str:
    """Get the weather for a city."""
    return city


class WeatherResponse(BaseModel):
    city: str


class MixedToolCallingModel(FakeToolCallingModel):
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
        if self.index == 0:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {}, "id": "weather"}],
                invalid_tool_calls=[
                    {
                        "name": "WeatherResponse",
                        "args": '{"city":',
                        "id": "structured",
                        "error": "Invalid JSON",
                    }
                ],
            )
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "WeatherResponse",
                        "args": {"city": "Paris"},
                        "id": "response",
                    }
                ],
            )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_create_agent_does_not_patch_model_output() -> None:
    model = InvalidToolCallingModel()
    agent = create_agent(model, [get_weather])

    result = agent.invoke({"messages": [HumanMessage("Weather?")]})

    assert model.index == 1
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)


def test_create_agent_answers_invalid_tool_calls_on_next_turn() -> None:
    model = InvalidToolCallingModel()
    agent = create_agent(model, [get_weather], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": [HumanMessage("Weather?")]}, config)
    model.invalid_tool_call_id = None
    result = agent.invoke({"messages": [HumanMessage("Try again")]}, config)

    tool_message = model.received_messages[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.name == "get_weather"
    assert tool_message.status == "error"
    assert "malformed or truncated" in tool_message.text
    assert [type(message) for message in result["messages"]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
        AIMessage,
    ]


async def test_create_agent_answers_invalid_tool_calls_async() -> None:
    model = InvalidToolCallingModel()
    agent = create_agent(model, [get_weather], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    await agent.ainvoke({"messages": [HumanMessage("Weather?")]}, config)
    model.invalid_tool_call_id = None
    result = await agent.ainvoke({"messages": [HumanMessage("Try again")]}, config)

    tool_message = model.received_messages[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.status == "error"
    assert isinstance(result["messages"][2], ToolMessage)


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


def test_invalid_structured_tool_call_does_not_end_agent() -> None:
    model = MixedToolCallingModel()
    agent = create_agent(
        model,
        [get_weather],
        response_format=ToolStrategy(WeatherResponse),
    )

    result = agent.invoke({"messages": [HumanMessage("Weather?")]})

    assert model.index == 2
    assert result["structured_response"] == WeatherResponse(city="Paris")
    answered = {
        message.tool_call_id: message.status
        for message in model.received_messages
        if isinstance(message, ToolMessage)
    }
    assert answered == {"structured": "error", "weather": "success"}


@pytest.mark.parametrize(("tool_call_id", "patched"), [(None, False), ("", True)])
def test_create_agent_patches_only_invalid_tool_calls_with_ids(
    tool_call_id: str | None, *, patched: bool
) -> None:
    model = InvalidToolCallingModel(invalid_tool_call_id=None)
    agent = create_agent(model, [get_weather])
    invalid_message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "get_weather",
                "args": '{"city":',
                "id": tool_call_id,
                "error": "Invalid JSON",
            }
        ],
    )

    agent.invoke({"messages": [HumanMessage("Weather?"), invalid_message]})

    assert isinstance(model.received_messages[-1], ToolMessage) is patched


def test_repair_promotes_invalid_tool_calls_into_serializable_tool_calls() -> None:
    """The repaired history must keep every answered id parented (#40853).

    Provider serializers disagree on `invalid_tool_calls`: langchain-anthropic
    drops them, so a synthetic `ToolMessage` answering one serialized to a
    `tool_result` with no `tool_use` parent and the Anthropic Messages API
    rejected the payload with a 400 - permanently, since the repair is persisted
    into state. After the repair the id lives in regular `tool_calls`, which
    every provider serializes.
    """
    model = InvalidToolCallingModel()
    agent = create_agent(model, [get_weather], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": [HumanMessage("Weather?")]}, config)
    model.invalid_tool_call_id = None
    agent.invoke({"messages": [HumanMessage("Try again")]}, config)

    repaired = model.received_messages[1]
    assert isinstance(repaired, AIMessage)
    assert [call["id"] for call in repaired.tool_calls] == ["call_1"]
    assert repaired.tool_calls[0]["name"] == "get_weather"
    # Truncated JSON cannot be recovered; the placeholder keeps the call
    # schema-parseable instead of leaking raw string args into serializers.
    assert repaired.tool_calls[0]["args"] == {}
    assert repaired.invalid_tool_calls == []


def test_repaired_history_pairs_every_tool_result_with_a_tool_call() -> None:
    """The provider-agnostic shape of the #40853 orphan check."""
    model = InvalidToolCallingModel()
    agent = create_agent(model, [get_weather], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": [HumanMessage("Weather?")]}, config)
    model.invalid_tool_call_id = None
    agent.invoke({"messages": [HumanMessage("Try again")]}, config)

    seen = model.received_messages
    tool_use_ids = {
        call["id"]
        for message in seen
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    tool_result_ids = {
        message.tool_call_id for message in seen if isinstance(message, ToolMessage)
    }
    assert tool_result_ids <= tool_use_ids


def test_repair_heals_persisted_threads_with_answered_invalid_calls() -> None:
    """A thread broken by the old repair must heal on the next model call."""
    model = InvalidToolCallingModel(invalid_tool_call_id=None)
    agent = create_agent(model, [get_weather])
    broken_history = [
        HumanMessage("Weather?"),
        AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "get_weather",
                    "args": '{"city":',
                    "id": "call_1",
                    "error": "Invalid JSON",
                }
            ],
        ),
        ToolMessage(
            "Tool call get_weather with id call_1 could not be executed",
            tool_call_id="call_1",
            status="error",
        ),
    ]

    agent.invoke({"messages": broken_history})

    seen = model.received_messages[1]
    assert isinstance(seen, AIMessage)
    assert [call["id"] for call in seen.tool_calls] == ["call_1"]
    assert seen.invalid_tool_calls == []


def test_repair_keeps_id_less_invalid_tool_calls_untouched() -> None:
    """Calls without an id get no ToolMessage, so promoting them would only
    add a parentless tool_call; they stay as invalid_tool_calls."""
    model = InvalidToolCallingModel(invalid_tool_call_id=None)
    agent = create_agent(model, [get_weather])
    invalid_message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "get_weather",
                "args": '{"city":',
                "id": None,
                "error": "Invalid JSON",
            }
        ],
    )

    agent.invoke({"messages": [HumanMessage("Weather?"), invalid_message]})

    seen = model.received_messages[1]
    assert isinstance(seen, AIMessage)
    assert seen.tool_calls == []
    assert len(seen.invalid_tool_calls) == 1
