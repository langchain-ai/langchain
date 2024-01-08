from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    See `ChatMessageHistory` for default implementation.

    Example:

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

    messages: List[BaseMessage]
    """A list of Messages stored in-memory."""

    def add_user_message(self, message: Union[HumanMessage, str]) -> None:
        """Convenience method for adding a human message string to the store.

        Deprecation Warning: This method will likely be deprecated in a future release.
            Code should use add_messages instead which can be made more efficient
            across implementations.

        Args:
            message: The human message to add
        """
        if isinstance(message, HumanMessage):
            self.add_message(message)
        else:
            self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: Union[AIMessage, str]) -> None:
        """Convenience method for adding an AI message string to the store.

        Deprecation Warning: This method will likely be deprecated in a future release.
            Code should use add_messages instead which can be made more efficient
            across implementations.

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
        """
        if type(self).add_messages != BaseChatMessageHistory.add_messages:
            # This means that the sub-class has implemented an efficient add_messages
            # method, so we should usage of add_message to that.
            self.add_messages([message])
        else:
            raise NotImplementedError(
                "add_message is not implemented for this class. "
                "Please implement add_message or add_messages."
            )

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages.

        Implementations should over-ride this method to handle bulk addition of messages
        in an efficient manner to avoid unnecessary round-trips to the underlying store.

        Args:
            messages: A list of BaseMessage objects to store.
        """
        for message in messages:
            self.add_message(message)

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""

    def __str__(self) -> str:
        """Return a string representation of the chat history."""
        return get_buffer_string(self.messages)
