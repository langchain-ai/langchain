from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
    RemoveMessage,
)
from langgraph.graph.message import add_messages


class TestHumanInTheLoopMiddlewareBeforeModel:
    """Test HumanInTheLoopMiddleware before_model behavior."""

    def test_first_message(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
        ]
        middleware = HumanInTheLoopMiddleware(interrupt_on={})
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 3
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1].type == "system"
        assert state_update["messages"][1].content == "You are a helpful assistant."
        assert state_update["messages"][2].type == "human"
        assert state_update["messages"][2].content == "Hello, how are you?"
        assert state_update["messages"][2].id == "2"

    def test_missing_tool_call(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[
                    ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})
                ],
                id="3",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="4"),
        ]
        middleware = HumanInTheLoopMiddleware(interrupt_on={})
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 6
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1] == input_messages[0]
        assert state_update["messages"][2] == input_messages[1]
        assert state_update["messages"][3] == input_messages[2]
        assert state_update["messages"][4].type == "tool"
        assert state_update["messages"][4].tool_call_id == "123"
        assert state_update["messages"][4].name == "get_events_for_days"
        assert state_update["messages"][5] == input_messages[3]
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 5
        assert updated_messages[0] == input_messages[0]
        assert updated_messages[1] == input_messages[1]
        assert updated_messages[2] == input_messages[2]
        assert updated_messages[3].type == "tool"
        assert updated_messages[3].tool_call_id == "123"
        assert updated_messages[3].name == "get_events_for_days"
        assert updated_messages[4] == input_messages[3]

    def test_no_missing_tool_calls(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[
                    ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})
                ],
                id="3",
            ),
            ToolMessage(content="I have no events for that date.", tool_call_id="123", id="4"),
            HumanMessage(content="What is the weather in Tokyo?", id="5"),
        ]
        middleware = HumanInTheLoopMiddleware(interrupt_on={})
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 6
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1:] == input_messages
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 5
        assert updated_messages == input_messages

    def test_two_missing_tool_calls(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[
                    ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})
                ],
                id="3",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="4"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[
                    ToolCall(id="456", name="get_events_for_days", args={"date_str": "2025-01-01"})
                ],
                id="5",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="6"),
        ]
        middleware = HumanInTheLoopMiddleware(interrupt_on={})
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 9
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1] == input_messages[0]
        assert state_update["messages"][2] == input_messages[1]
        assert state_update["messages"][3] == input_messages[2]
        assert state_update["messages"][4].type == "tool"
        assert state_update["messages"][4].tool_call_id == "123"
        assert state_update["messages"][4].name == "get_events_for_days"
        assert state_update["messages"][5] == input_messages[3]
        assert state_update["messages"][6] == input_messages[4]
        assert state_update["messages"][7].type == "tool"
        assert state_update["messages"][7].tool_call_id == "456"
        assert state_update["messages"][7].name == "get_events_for_days"
        assert state_update["messages"][8] == input_messages[5]
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 8
        assert updated_messages[0] == input_messages[0]
        assert updated_messages[1] == input_messages[1]
        assert updated_messages[2] == input_messages[2]
        assert updated_messages[3].type == "tool"
        assert updated_messages[3].tool_call_id == "123"
        assert updated_messages[3].name == "get_events_for_days"
        assert updated_messages[4] == input_messages[3]
        assert updated_messages[5] == input_messages[4]
        assert updated_messages[6].type == "tool"
        assert updated_messages[6].tool_call_id == "456"
        assert updated_messages[6].name == "get_events_for_days"
        assert updated_messages[7] == input_messages[5]
