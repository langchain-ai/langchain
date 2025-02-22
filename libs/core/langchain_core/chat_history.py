"""**Chat message history** stores a history of the message interactions in a chat.

**Class hierarchy:**

.. code-block::

    BaseChatMessageHistory --> <name>ChatMessageHistory  # Examples: FileChatMessageHistory, PostgresChatMessageHistory

**Main helpers:**

.. code-block::

    AIMessage, HumanMessage, BaseMessage

"""  # noqa: E501

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union

from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    Implementations guidelines:

    Implementations are expected to over-ride all or some of the following methods:

    * add_messages: sync variant for bulk addition of messages
    * aadd_messages: async variant for bulk addition of messages
    * messages: sync variant for getting messages
    * aget_messages: async variant for getting messages
    * clear: sync variant for clearing messages
    * aclear: async variant for clearing messages

    add_messages contains a default implementation that calls add_message
    for each message in the sequence. This is provided for backwards compatibility
    with existing implementations which only had add_message.

    Async variants all have default implementations that call the sync variants.
    Implementers can choose to over-ride the async implementations to provide
    truly async implementations.

    Usage guidelines:

    When used for updating history, users should favor usage of `add_messages`
    over `add_message` or other variants like `add_user_message` and `add_ai_message`
    to avoid unnecessary round-trips to the underlying persistence layer.

    Example: Shows a default implementation.

        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_messages(self, messages: Sequence[BaseMessage]) -> None:
                   all_messages = list(self.messages) # Existing messages
                   all_messages.extend(messages) # Add new messages

                   serialized = [message_to_dict(message) for message in all_messages]
                   # Can be further optimized by only writing new messages
                   # using append mode.
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: list[BaseMessage]
    """A property or attribute that returns a list of messages.

    In general, getting the messages may involve IO to the underlying
    persistence layer, so this operation is expected to incur some
    latency.
    """

    async def aget_messages(self) -> list[BaseMessage]:
        """Async version of getting messages.

        Can over-ride this method to provide an efficient async implementation.

        In general, fetching messages may involve IO to the underlying
        persistence layer.
        """
        from langchain_core.runnables.config import run_in_executor

        return await run_in_executor(None, lambda: self.messages)

    def add_user_message(self, message: Union[HumanMessage, str]) -> None:
        """Convenience method for adding a human message string to the store.

        Please note that this is a convenience method. Code should favor the
        bulk add_messages interface instead to save on round-trips to the underlying
        persistence layer.

        This method may be deprecated in a future release.

        Args:
            message: The human message to add to the store.
        """
        if isinstance(message, HumanMessage):
            self.add_message(message)
        else:
            self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: Union[AIMessage, str]) -> None:
        """Convenience method for adding an AI message string to the store.

        Please note that this is a convenience method. Code should favor the bulk
        add_messages interface instead to save on round-trips to the underlying
        persistence layer.

        This method may be deprecated in a future release.

        Args:
            message: The AI message to add.
        """
        if isinstance(message, AIMessage):
            self.add_message(message)
        else:
            self.add_message(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.

        Raises:
            NotImplementedError: If the sub-class has not implemented an efficient
                add_messages method.
        """
        if type(self).add_messages != BaseChatMessageHistory.add_messages:
            # This means that the sub-class has implemented an efficient add_messages
            # method, so we should use it.
            self.add_messages([message])
        else:
            msg = (
                "add_message is not implemented for this class. "
                "Please implement add_message or add_messages."
            )
            raise NotImplementedError(msg)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add a list of messages.

        Implementations should over-ride this method to handle bulk addition of messages
        in an efficient manner to avoid unnecessary round-trips to the underlying store.

        Args:
            messages: A sequence of BaseMessage objects to store.
        """
        for message in messages:
            self.add_message(message)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Async add a list of messages.

        Args:
            messages: A sequence of BaseMessage objects to store.
        """
        from langchain_core.runnables.config import run_in_executor

        await run_in_executor(None, self.add_messages, messages)

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store."""

    async def aclear(self) -> None:
        """Async remove all messages from the store."""
        from langchain_core.runnables.config import run_in_executor

        await run_in_executor(None, self.clear)

    def __str__(self) -> str:
        """Return a string representation of the chat history."""
        return get_buffer_string(self.messages)


class InMemoryChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in a memory list.
    """

    messages: list[BaseMessage] = Field(default_factory=list)
    """A list of messages stored in memory."""

    async def aget_messages(self) -> list[BaseMessage]:
        """Async version of getting messages.

        Can over-ride this method to provide an efficient async implementation.
        In general, fetching messages may involve IO to the underlying
        persistence layer.

        Returns:
            List of messages.
        """
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store.

        Args:
            message: The message to add.
        """
        self.messages.append(message)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Async add messages to the store.

        Args:
            messages: The messages to add.
        """
        self.add_messages(messages)

    def clear(self) -> None:
        """Clear all messages from the store."""
        self.messages = []

    async def aclear(self) -> None:
        """Async clear all messages from the store."""
        self.clear()
