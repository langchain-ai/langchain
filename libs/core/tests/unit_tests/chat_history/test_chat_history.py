from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage


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
            raise NotImplementedError()

    store: list[BaseMessage] = []
    chat_history = SampleChatHistory(store=store)
    chat_history.add_message(HumanMessage(content="Hello"))
    assert len(store) == 1
    assert store[0] == HumanMessage(content="Hello")
    chat_history.add_message(HumanMessage(content="World"))
    assert len(store) == 2
    assert store[1] == HumanMessage(content="World")

    chat_history.add_messages(
        [HumanMessage(content="Hello"), HumanMessage(content="World")]
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
            raise NotImplementedError()

    chat_history = BulkAddHistory(store=store)
    chat_history.add_message(HumanMessage(content="Hello"))
    assert len(store) == 1
    assert store[0] == HumanMessage(content="Hello")
    chat_history.add_message(HumanMessage(content="World"))
    assert len(store) == 2
    assert store[1] == HumanMessage(content="World")

    chat_history.add_messages(
        [HumanMessage(content="Hello"), HumanMessage(content="World")]
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
        [HumanMessage(content="Hello"), HumanMessage(content="World")]
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
