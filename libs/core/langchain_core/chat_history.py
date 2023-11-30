from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, List, Union, cast

from langchain_core.load.load import loads
from langchain_core.load.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedConstructorMemory,
    SerializedNotImplemented,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    message_to_dict,
    messages_from_dict,
)


class BaseChatMessageHistory(Serializable, ABC):
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

    def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
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

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return True

    def to_json(
        self
    ) -> Union[
        SerializedConstructor, SerializedNotImplemented, SerializedConstructorMemory
    ]:
        super_serialized: Union[
            SerializedConstructor, SerializedNotImplemented
        ] = super().to_json()
        serialized: SerializedConstructorMemory = SerializedConstructorMemory(
            lc=super_serialized.get("lc", 0),
            id=super_serialized.get("id", []),
            type="constructor",
            kwargs=cast(dict, super_serialized.get("kwargs", {})),
            obj=None,
            repr=str(super_serialized.get("repr", "")),
        )
        serialized["obj"] = json.loads(
            json.dumps(
                self,
                default=lambda o: message_to_dict(o)
                if isinstance(o, BaseMessage)
                else o.__dict__,
                sort_keys=True,
                indent=4,
            )
        )

        return serialized

    @classmethod
    def from_json(cls, json_input: str) -> Any:
        deserialized = loads(json_input)

        memory_dict = json.loads(json_input)

        messages = messages_from_dict(memory_dict["obj"]["messages"])

        # Extract additional attributes from memory_dict
        additional_attributes = {
            key: memory_dict[key] for key in memory_dict["obj"] if key != "messages"
        }

        deserialized.messages = messages

        deserialized.__dict__.update(additional_attributes)

        return deserialized
