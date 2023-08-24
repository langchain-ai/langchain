import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional, Union

from bs4 import BeautifulSoup

from langchain import schema
from langchain.chat_loaders import base as chat_loaders

logger = logging.getLogger(__name__)


class TelegramChatLoader(chat_loaders.BaseChatLoader):
    def __init__(
        self,
        path: Union[str, Path],
        user_name: Optional[str] = None,
        merge_runs: bool = True,
    ):
        """
        Initialize the chat loader with the path to the exported Telegram chat file
        or directory.

        To export, use the Telegram Desktop app from https://desktop.telegram.org/,
        select a conversation, click the three dots in the top right corner, and
        select "Export chat history". Then select "Machine-readable JSON"
        (preferred) to export.

        Note: the 'lite' versions of the desktop app (like "Telegram for MacOS")
        do not support exporting chat history.

        :param path: Path to the exported Telegram chat zip, directory, json,
            or HTML file.
        :param user_name: Name of the user who will be mapped to the "AI" role.
        :param merge_runs: Whether to merge message 'runs' into a single message.
            A message run is a sequence of messages from the same sender.
        """
        self.path = path if isinstance(path, str) else str(path)
        self.user_name = user_name
        self.merge_runs = merge_runs

    def _load_single_chat_session_html(
        self, file_path: str
    ) -> chat_loaders.ChatSession:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        results: List[Union[schema.HumanMessage, schema.AIMessage]] = []
        previous_sender = None
        for message in soup.select(".message.default"):
            timestamp = message.select_one(".pull_right.date.details")["title"]
            from_name_element = message.select_one(".from_name")
            if from_name_element is None and previous_sender is None:
                logger.debug("from_name not found in message")
                continue
            elif from_name_element is None:
                from_name = previous_sender
            else:
                from_name = from_name_element.text.strip()
            text = message.select_one(".text").text.strip()

            if from_name == previous_sender and self.merge_runs:
                results[-1].content += "\n\n" + text
                results[-1].additional_kwargs["events"].append(
                    {"message_time": timestamp}
                )
            else:
                message_cls = (
                    schema.AIMessage
                    if from_name == self.user_name
                    else schema.HumanMessage
                )
                results.append(
                    message_cls(
                        content=text,
                        additional_kwargs={
                            "sender": from_name,
                            "events": [{"message_time": timestamp}],
                        },
                    )
                )
            previous_sender = from_name

        return chat_loaders.ChatSession(messages=results)

    def _load_single_chat_session_json(
        self, file_path: str
    ) -> chat_loaders.ChatSession:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        messages = data.get("messages", [])
        results: List[Union[schema.AIMessage, schema.HumanMessage]] = []
        previous_sender = None
        for message in messages:
            text = message.get("text", "")
            timestamp = message.get("date", "")
            from_name = message.get("from", "")

            if from_name == previous_sender and self.merge_runs:
                results[-1].content += "\n\n" + text
                results[-1].additional_kwargs["events"].append(
                    {"message_time": timestamp}
                )
            else:
                message_cls = (
                    schema.AIMessage
                    if from_name == self.user_name
                    else schema.HumanMessage
                )
                results.append(
                    message_cls(
                        content=text,
                        additional_kwargs={
                            "sender": from_name,
                            "events": [{"message_time": timestamp}],
                        },
                    )
                )
            previous_sender = from_name

        return chat_loaders.ChatSession(messages=results)

    def _iterate_files(self, path: str) -> Iterator[str]:
        if os.path.isfile(path) and path.endswith((".html", ".json")):
            yield path
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith((".html", ".json")):
                        yield os.path.join(root, file)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zip_file:
                for file in zip_file.namelist():
                    if file.endswith((".html", ".json")):
                        yield zip_file.extract(file)

    def lazy_load(self) -> Iterator[chat_loaders.ChatSession]:
        """
        Lazy load the messages from the chat file and yield them in the required format.

        :return: Iterator of chat sessions containing messages.
        """
        for file_path in self._iterate_files(self.path):
            if file_path.endswith(".html"):
                yield self._load_single_chat_session_html(file_path)
            elif file_path.endswith(".json"):
                yield self._load_single_chat_session_json(file_path)
