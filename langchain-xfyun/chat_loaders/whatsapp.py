import logging
import os
import re
import zipfile
from typing import Iterator, List, Union

from langchain import schema
from langchain.chat_loaders import base as chat_loaders
from langchain.schema import messages

logger = logging.getLogger(__name__)


class WhatsAppChatLoader(chat_loaders.BaseChatLoader):
    def __init__(self, path: str):
        """Initialize the WhatsAppChatLoader.

        Args:
            path (str): Path to the exported WhatsApp chat
                zip directory, folder, or file.

        To generate the dump, open the chat, click the three dots in the top
        right corner, and select "More". Then select "Export chat" and
        choose "Without media".
        """
        self.path = path
        ignore_lines = [
            "This message was deleted",
            "<Media omitted>",
            "image omitted",
            "Messages and calls are end-to-end encrypted. No one outside of this chat,"
            " not even WhatsApp, can read or listen to them.",
        ]
        self._ignore_lines = re.compile(
            r"(" + "|".join([r"\u200E*" + line for line in ignore_lines]) + r")",
            flags=re.IGNORECASE,
        )
        self._message_line_regex = re.compile(
            r"\u200E*\[?(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2} (?:AM|PM))\]?[ \u200E]*([^:]+): (.+)",  # noqa
            flags=re.IGNORECASE,
        )

    def _load_single_chat_session(self, file_path: str) -> chat_loaders.ChatSession:
        """Load a single chat session from a file.

        Args:
            file_path (str): Path to the chat file.

        Returns:
            ChatSession: The loaded chat session.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            txt = file.read()

        # Split messages by newlines, but keep multi-line messages grouped
        chat_lines: List[str] = []
        current_message = ""
        for line in txt.split("\n"):
            if self._message_line_regex.match(line):
                if current_message:
                    chat_lines.append(current_message)
                current_message = line
            else:
                current_message += " " + line.strip()
        if current_message:
            chat_lines.append(current_message)
        results: List[Union[messages.HumanMessage, messages.AIMessage]] = []
        for line in chat_lines:
            result = self._message_line_regex.match(line.strip())
            if result:
                timestamp, sender, text = result.groups()
                if not self._ignore_lines.match(text.strip()):
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
            else:
                logger.debug(f"Could not parse line: {line}")
        return chat_loaders.ChatSession(messages=results)

    def _iterate_files(self, path: str) -> Iterator[str]:
        """Iterate over the files in a directory or zip file.

        Args:
            path (str): Path to the directory or zip file.

        Yields:
            str: The path to each file.
        """
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".txt"):
                        yield os.path.join(root, file)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zip_file:
                for file in zip_file.namelist():
                    if file.endswith(".txt"):
                        yield zip_file.extract(file)

    def lazy_load(self) -> Iterator[chat_loaders.ChatSession]:
        """Lazy load the messages from the chat file and yield
        them as chat sessions.

        Yields:
            Iterator[ChatSession]: The loaded chat sessions.
        """
        yield self._load_single_chat_session(self.path)
