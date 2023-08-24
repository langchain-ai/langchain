import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Iterator, List, Union

from langchain import schema
from langchain.chat_loaders import base as chat_loaders

logger = logging.getLogger(__name__)


class TelegramChatLoader(chat_loaders.BaseChatLoader):
    """A loading utility for converting telegram conversations
    to LangChain chat messages.

    To export, use the Telegram Desktop app from
    https://desktop.telegram.org/, select a conversation, click the three dots
    in the top right corner, and select "Export chat history". Then select
    "Machine-readable JSON" (preferred) to export. Note: the 'lite' versions of
    the desktop app (like "Telegram for MacOS") do not support exporting chat
    history.
    """

    def __init__(
        self,
        path: Union[str, Path],
    ):
        """Initialize the TelegramChatLoader.

        Args:
            path (Union[str, Path]): Path to the exported Telegram chat zip,
                 directory, json, or HTML file.
        """
        self.path = path if isinstance(path, str) else str(path)

    def _load_single_chat_session_html(
        self, file_path: str
    ) -> chat_loaders.ChatSession:
        """Load a single chat session from an HTML file.

        Args:
            file_path (str): Path to the HTML file.

        Returns:
            chat_loaders.ChatSession: The loaded chat session.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Please install the 'beautifulsoup4' package to load"
                " Telegram HTML files. You can do this by running"
                "'pip install beautifulsoup4' in your terminal."
            )
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
            results.append(
                schema.HumanMessage(
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
        """Load a single chat session from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            chat_loaders.ChatSession: The loaded chat session.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        messages = data.get("messages", [])
        results: List[schema.BaseMessage] = []
        for message in messages:
            text = message.get("text", "")
            timestamp = message.get("date", "")
            from_name = message.get("from", "")

            results.append(
                schema.HumanMessage(
                    content=text,
                    additional_kwargs={
                        "sender": from_name,
                        "events": [{"message_time": timestamp}],
                    },
                )
            )

        return chat_loaders.ChatSession(messages=results)

    def _iterate_files(self, path: str) -> Iterator[str]:
        """Iterate over files in a directory or zip file.

        Args:
            path (str): Path to the directory or zip file.

        Yields:
            str: Path to each file.
        """
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
        """Lazy load the messages from the chat file and yield them
        in as chat sessions.

        Yields:
            chat_loaders.ChatSession: The loaded chat session.
        """
        for file_path in self._iterate_files(self.path):
            if file_path.endswith(".html"):
                yield self._load_single_chat_session_html(file_path)
            elif file_path.endswith(".json"):
                yield self._load_single_chat_session_json(file_path)
