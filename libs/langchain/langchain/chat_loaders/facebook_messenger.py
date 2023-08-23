from langchain.chat_loaders.base import BaseChatLoader
from pathlib import Path
from typing import Iterator, List
from langchain.schema.messages import BaseMessage
from langchain.schema.messages import HumanMessage, AIMessage
import json
import logging

logger = logging.getLogger(__file__)


class InvalidFile(Exception):
    """To raise when file cannot be loaded."""

class SingleFileFacebookMessengerChatLoader(BaseChatLoader):
    file_path: Path
    person_name: str

    def lazy_load_messages(self) -> Iterator[List[BaseMessage]]:
        with open(self.file_path) as f:
            data = json.load(f)
        if len(data['participants']) != 2:
            raise InvalidFile("Only works on data with two participants")
        names = [p['name'] for p in data['participants']]
        if self.person_name not in names:
            raise InvalidFile(f"Sender name {self.person_name} not found in participants")
        sorted_data = sorted(data['messages'], key=lambda x: x['timestamp_ms'])
        messages = []
        for m in sorted_data:
            if m['sender_name'] == self.person_name:
                message_class = AIMessage
            else:
                message_class = HumanMessage
            messages.append(message_class(content=m['content']))
        yield [messages]


class FolderFacebookMessengerChatLoader(BaseChatLoader):
    directory_path: Path
    person_name: str

    def lazy_load_messages(self) -> Iterator[List[BaseMessage]]:
        inbox_path = self.directory_path / "inbox"
        for _dir in inbox_path.iterdir():
            if _dir.is_dir():
                for _file in _dir.iterdir():
                    if _file.suffix.lower() == ".json":
                        file_loader = SingleFileFacebookMessengerChatLoader(
                            file_path = _file,
                            person_name = self.person_name
                        )
                        try:
                            for result in  file_loader.lazy_load_messages():
                                yield result
                        except InvalidFile:
                            logger.info(f"Could not load file: {_file}")
