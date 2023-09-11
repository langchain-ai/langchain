import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Union

from langchain import schema
from langchain.chat_loaders.base import BaseChatLoader, ChatSession

logger = logging.getLogger(__name__)


class SlackChatLoader(BaseChatLoader):
    """Load `Slack` conversations from a dump zip file."""

    def __init__(
        self,
        path: Union[str, Path],
    ):
        """
        Initialize the chat loader with the path to the exported Slack dump zip file.

        :param path: Path to the exported Slack dump zip file.
        """
        self.zip_path = path if isinstance(path, Path) else Path(path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"File {self.zip_path} not found")

    def _load_single_chat_session(self, messages: List[Dict]) -> ChatSession:
        results: List[Union[schema.AIMessage, schema.HumanMessage]] = []
        previous_sender = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = message.get("text", "")
            timestamp = message.get("ts", "")
            sender = message.get("user", "")
            if not sender:
                continue
            skip_pattern = re.compile(
                r"<@U\d+> has joined the channel", flags=re.IGNORECASE
            )
            if skip_pattern.match(text):
                continue
            if sender == previous_sender:
                results[-1].content += "\n\n" + text
                results[-1].additional_kwargs["events"].append(
                    {"message_time": timestamp}
                )
            else:
                results.append(
                    schema.HumanMessage(
                        role=sender,
                        content=text,
                        additional_kwargs={
                            "sender": sender,
                            "events": [{"message_time": timestamp}],
                        },
                    )
                )
            previous_sender = sender
        return ChatSession(messages=results)

    def _read_json(self, zip_file: zipfile.ZipFile, file_path: str) -> List[dict]:
        """Read JSON data from a zip subfile."""
        with zip_file.open(file_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list of dictionaries, got {type(data)}")
        return data

    def lazy_load(self) -> Iterator[ChatSession]:
        """
        Lazy load the chat sessions from the Slack dump file and yield them
        in the required format.

        :return: Iterator of chat sessions containing messages.
        """
        with zipfile.ZipFile(str(self.zip_path), "r") as zip_file:
            for file_path in zip_file.namelist():
                if file_path.endswith(".json"):
                    messages = self._read_json(zip_file, file_path)
                    yield self._load_single_chat_session(messages)
