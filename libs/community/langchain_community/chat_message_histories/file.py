import json
from pathlib import Path
from typing import List, Optional

from langchain_core.chat_history import (
    BaseChatMessageHistory,
)
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in a local file."""

    def __init__(
        self,
        file_path: str,
        *,
        encoding: Optional[str] = None,
        ensure_ascii: bool = True,
    ) -> None:
        """Initialize the file path for the chat history.
        Args:
            file_path: The path to the local file to store the chat history.
            encoding: The encoding to use for file operations. Defaults to None.
            ensure_ascii: If True, escape non-ASCII in JSON. Defaults to True.
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii

        if not self.file_path.exists():
            self.file_path.touch()
            self.file_path.write_text(
                json.dumps([], ensure_ascii=self.ensure_ascii), encoding=self.encoding
            )

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from the local file"""
        items = json.loads(self.file_path.read_text(encoding=self.encoding))
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in the local file"""
        messages = messages_to_dict(self.messages)
        messages.append(messages_to_dict([message])[0])
        self.file_path.write_text(
            json.dumps(messages, ensure_ascii=self.ensure_ascii), encoding=self.encoding
        )

    def clear(self) -> None:
        """Clear session memory from the local file"""
        self.file_path.write_text(
            json.dumps([], ensure_ascii=self.ensure_ascii), encoding=self.encoding
        )
