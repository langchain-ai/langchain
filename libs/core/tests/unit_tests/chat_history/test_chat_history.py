from collections.abc import Sequence

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.tool import ToolCall


def test_add_message_implementation_only() -> None:
    """Test implementation of add_message only."""

    class SampleChatHistory(BaseChatMessageHistory):
        def __init__(self, *, store: list[BaseMessage]) -> None:
            self.store = store

        def add_message(self, message: BaseMessage) -> None:
            """Add a message to the store."""
            self.store.append(message)

        def clear(self) -> None:
            """Clear the store."""
            raise NotImplementedError

    store: list[BaseMessage] = []
    chat_history = SampleChatHistory(store=store)
    chat_history.add_message(HumanMessage(content="Hello"))
    assert len(store) == 1
    assert store[0] == HumanMessage(content="Hello")
    chat_history.add_message(HumanMessage(content="World"))
    assert len(store) == 2
    assert store[1] == HumanMessage(content="World")

    chat_history.add_messages(
        [
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
    )
    assert len(store) == 4
    assert store[2] == HumanMessage(content="Hello")
    assert store[3] == HumanMessage(content="World")


def test_bulk_message_implementation_only() -> None:
    """Test that SampleChatHistory works as expected."""
    store: list[BaseMessage] = []

    class BulkAddHistory(BaseChatMessageHistory):
        def __init__(self, *, store: list[BaseMessage]) -> None:
            self.store = store

        def add_messages(self, message: Sequence[BaseMessage]) -> None:
            """Add a message to the store."""
            self.store.extend(message)

        def clear(self) -> None:
            """Clear the store."""
            raise NotImplementedError

    chat_history = BulkAddHistory(store=store)
    chat_history.add_message(HumanMessage(content="Hello"))
    assert len(store) == 1
    assert store[0] == HumanMessage(content="Hello")
    chat_history.add_message(HumanMessage(content="World"))
    assert len(store) == 2
    assert store[1] == HumanMessage(content="World")

    chat_history.add_messages(
        [
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
    )
    assert len(store) == 4
    assert store[2] == HumanMessage(content="Hello")
    assert store[3] == HumanMessage(content="World")


async def test_async_interface() -> None:
    """Test async interface for BaseChatMessageHistory."""

    class BulkAddHistory(BaseChatMessageHistory):
        def __init__(self) -> None:
            self.messages = []

        def add_messages(self, message: Sequence[BaseMessage]) -> None:
            """Add a message to the store."""
            self.messages.extend(message)

        def clear(self) -> None:
            """Clear the store."""
            self.messages.clear()

    chat_history = BulkAddHistory()
    await chat_history.aadd_messages(
        [
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
    )
    assert await chat_history.aget_messages() == [
        HumanMessage(content="Hello"),
        HumanMessage(content="World"),
    ]
    await chat_history.aadd_messages([HumanMessage(content="!")])
    assert await chat_history.aget_messages() == [
        HumanMessage(content="Hello"),
        HumanMessage(content="World"),
        HumanMessage(content="!"),
    ]
    await chat_history.aclear()
    assert await chat_history.aget_messages() == []


def test_inmemory_chat_history_serialization_preserves_tool_calls() -> None:
    """Test that AIMessage.tool_calls is preserved during serialization.

    This is a regression test for:
    https://github.com/langchain-ai/langchain/issues/34925

    The issue was that when using `model_dump_json()` on InMemoryChatMessageHistory,
    subclass-specific fields like `tool_calls` on AIMessage were silently dropped
    because the `messages` field was typed as `list[BaseMessage]` without
    `SerializeAsAny`.
    """
    tool_call = ToolCall(name="test_tool", args={"arg1": "value1"}, id="call_123")
    ai_message = AIMessage(content="Using a tool", tool_calls=[tool_call])

    chat_history = InMemoryChatMessageHistory()
    chat_history.add_message(ai_message)

    # Serialize and deserialize
    json_data = chat_history.model_dump_json()
    restored = InMemoryChatMessageHistory.model_validate_json(json_data)

    # Verify the message is restored correctly
    assert len(restored.messages) == 1
    restored_msg = restored.messages[0]
    assert isinstance(restored_msg, AIMessage)
    assert restored_msg.content == "Using a tool"

    # The key assertion: tool_calls should be preserved
    assert len(restored_msg.tool_calls) == 1
    assert restored_msg.tool_calls[0]["name"] == "test_tool"
    assert restored_msg.tool_calls[0]["args"] == {"arg1": "value1"}
    assert restored_msg.tool_calls[0]["id"] == "call_123"


def test_inmemory_chat_history_serialization_preserves_response_metadata() -> None:
    """Test that AIMessage.response_metadata is preserved during serialization."""
    ai_message = AIMessage(
        content="Hello",
        response_metadata={"model": "gpt-4", "finish_reason": "stop"},
    )

    chat_history = InMemoryChatMessageHistory()
    chat_history.add_message(ai_message)

    json_data = chat_history.model_dump_json()
    restored = InMemoryChatMessageHistory.model_validate_json(json_data)

    restored_msg = restored.messages[0]
    assert isinstance(restored_msg, AIMessage)
    assert restored_msg.response_metadata == {"model": "gpt-4", "finish_reason": "stop"}
