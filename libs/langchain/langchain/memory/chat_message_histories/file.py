"""Chat message history that stores history in a local file.

This chat history is mainly useful for testing / prototyping purposes.
"""
import json
import logging
import re
from pathlib import Path
from typing import Callable, List, Type, TypeVar, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)

Self = TypeVar("Self", bound="FileChatMessageHistory")


def _is_valid_session_id(session_id: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(session_id))


class FileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in a local file.

    Examples:

        .. code-block:: python

            from langchain_core.messages import HumanMessage

            from langchain.memory import FileChatMessageHistory

            history = FileHumanMessageHistory("history.json")
            history.add_message(HumanMessage("Hello, world!"))
            history.messages # [HumanMessage("Hello, world!")]
            history.clear()
            history.messages # []
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        """Chat message history that stores history in a local file.

        Args:
            file_path: path of the local file to store the messages
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            file_path.touch()
            file_path.write_text(json.dumps([]))

        self.file_path = file_path

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

    @classmethod
    def create_session_factory(
        cls: Type[Self], base_dir: Union[str, Path]
    ) -> Callable[[str], Self]:
        """Create a session ID factory that creates session IDs from a base dir.

        Args:
            base_dir: Base directory to use for storing the chat histories.

        Returns:
            A session ID factory that creates session IDs from a base path.
        """
        base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir

        def get_chat_history(session_id: str) -> Self:
            """Get a chat history from a session ID."""
            if not _is_valid_session_id(session_id):
                raise ValueError(
                    f"Session ID {session_id} is not in a valid format. "
                    "Session ID must only contain alphanumeric characters, "
                    "hyphens, and underscores."
                )

            file_path = base_dir_ / f"{session_id}.json"
            return cls(file_path)

        return get_chat_history
