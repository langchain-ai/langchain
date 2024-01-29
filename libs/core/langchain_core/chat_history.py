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

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
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

        Args:
            message: The human message to add
        """
        if isinstance(message, HumanMessage):
            self.add_message(message)
        else:
            self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: Union[AIMessage, str]) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The AI message to add.
        """
        if isinstance(message, AIMessage):
            self.add_message(message)
        else:
            self.add_message(AIMessage(content=message))

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""

    def __str__(self) -> str:
        return get_buffer_string(self.messages)
