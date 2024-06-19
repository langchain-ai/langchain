import json
from pathlib import Path
from typing import List, Optional

from langchain_core.chat_history import (
    BaseChatMessageHistory,
)
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in a local file."""

    def __init__(self, file_path: str, *, history_size: Optional[int] = None) -> None:
        """Initialize the file path for the chat history.

        Args:
            file_path: The path to the local file to store the chat history.
            history_size: Maximum number fo messages to retrieve. If None then
                there is no limit. If not None then only the latest `history_size`
                messages are retrieved.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.file_path.touch()
            self.file_path.write_text(json.dumps([]))
        self.history_size = history_size

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from the local file"""
        items = json.loads(self.file_path.read_text())
        if self.history_size:
            items = items[-self.history_size :]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in the local file"""
        messages = messages_to_dict(self.messages)
        messages.append(messages_to_dict([message])[0])
        self.file_path.write_text(json.dumps(messages))

    def clear(self) -> None:
        """Clear session memory from the local file"""
        self.file_path.write_text(json.dumps([]))
