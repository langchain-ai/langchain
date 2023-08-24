import json
import logging
from pathlib import Path
from typing import Iterator, Union

from langchain.chat_loaders.base import BaseChatLoader, ChatSession
from langchain.schema.messages import AIMessage, HumanMessage

logger = logging.getLogger(__file__)


class InvalidFile(Exception):
    """To raise when file cannot be loaded."""


class SingleFileFacebookMessengerChatLoader(BaseChatLoader):
    def __init__(self, file_path: Union[Path, str], person_name: str) -> None:
        super().__init__()
        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.person_name = person_name

    def lazy_load(self) -> Iterator[ChatSession]:
        with open(self.file_path) as f:
            data = json.load(f)
        if len(data["participants"]) != 2:
            raise InvalidFile("Only works on data with two participants")
        names = [p["name"] for p in data["participants"]]
        if self.person_name not in names:
            raise InvalidFile(
                f"Sender name {self.person_name} not found in participants"
            )
        sorted_data = sorted(data["messages"], key=lambda x: x["timestamp_ms"])
        messages = []
        for m in sorted_data:
            if m["sender_name"] == self.person_name:
                message_class = AIMessage
            else:
                message_class = HumanMessage
            messages.append(
                message_class(
                    content=m["content"], extra_kwargs={"sender": m["sender_name"]}
                )
            )
        yield ChatSession(messages=messages)


class FolderFacebookMessengerChatLoader(BaseChatLoader):
    def __init__(self, directory_path: Union[str, Path], person_name: str) -> None:
        super().__init__()
        self.directory_path = (
            Path(directory_path) if isinstance(directory_path, str) else directory_path
        )
        self.person_name = person_name

    def lazy_load(self) -> Iterator[ChatSession]:
        inbox_path = self.directory_path / "inbox"
        for _dir in inbox_path.iterdir():
            if _dir.is_dir():
                for _file in _dir.iterdir():
                    if _file.suffix.lower() == ".json":
                        file_loader = SingleFileFacebookMessengerChatLoader(
                            file_path=_file, person_name=self.person_name
                        )
                        try:
                            for result in file_loader.lazy_load():
                                yield result
                        except InvalidFile:
                            logger.info(f"Could not load file: {_file}")
