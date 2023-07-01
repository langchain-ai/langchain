import json
import logging
from pathlib import Path
from typing import List

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, messages_from_dict, messages_to_dict

logger = logging.getLogger(__name__)


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that stores history in a local file.

    Args:
        file_path: path of the local file to store the messages.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.file_path.touch()
            self.file_path.write_text(json.dumps([]))

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from the local file"""
        items = json.loads(self.file_path.read_text())
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
