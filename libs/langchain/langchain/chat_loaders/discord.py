"""Module for loading chat messages from a Discord chat.

This module provides a class, `DiscordChatLoader`, that can be used to
load chat messages from a Discord chat. The chat messages can be copied
from the Discord app and pasted into a text file, which can then be
passed to the `DiscordChatLoader` for loading.

Example usage:

```python
loader = DiscordChatLoader(path="path/to/chat.txt")
sessions = loader.lazy_load()
for session in sessions:
    for message in session.messages:
        print(message.content)
```
"""

import logging
import re
from typing import Iterator, List

from langchain import schema
from langchain.chat_loaders import base as chat_loaders

logger = logging.getLogger(__name__)


class DiscordChatLoader(chat_loaders.BaseChatLoader):
    def __init__(self, path: str):
        """
        Initialize the Discord chat loader.

        Args:
            path: Path to the exported Discord chat text file.
        """
        self.path = path
        self._message_line_regex = re.compile(
            r"(.+?) — (\w{3,9} \d{1,2}(?:st|nd|rd|th)?(?:, \d{4})? \d{1,2}:\d{2} (?:AM|PM)|Today at \d{1,2}:\d{2} (?:AM|PM)|Yesterday at \d{1,2}:\d{2} (?:AM|PM))",  # noqa
            flags=re.DOTALL,
        )

    def _load_single_chat_session_from_txt(
        self, file_path: str
    ) -> chat_loaders.ChatSession:
        """
        Load a single chat session from a text file.

        Args:
            file_path: Path to the text file containing the chat messages.

        Returns:
            A `ChatSession` object containing the loaded chat messages.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        results: List[schema.BaseMessage] = []
        current_sender = None
        current_timestamp = None
        current_content = []
        for line in lines:
            if re.match(
                r".+? — (\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} (?:AM|PM)|Today at \d{1,2}:\d{2} (?:AM|PM)|Yesterday at \d{1,2}:\d{2} (?:AM|PM))",  # noqa
                line,
            ):
                if current_sender and current_content:
                    results.append(
                        schema.HumanMessage(
                            content="".join(current_content).strip(),
                            additional_kwargs={
                                "sender": current_sender,
                                "events": [{"message_time": current_timestamp}],
                            },
                        )
                    )
                current_sender, current_timestamp = line.split(" — ")[:2]
                current_content = [
                    line[len(current_sender) + len(current_timestamp) + 4 :].strip()
                ]
            elif re.match(r"\[\d{1,2}:\d{2} (?:AM|PM)\]", line.strip()):
                results.append(
                    schema.HumanMessage(
                        content="".join(current_content).strip(),
                        additional_kwargs={
                            "sender": current_sender,
                            "events": [{"message_time": current_timestamp}],
                        },
                    )
                )
                current_timestamp = line.strip()[1:-1]
                current_content = []
            else:
                current_content.append("\n" + line.strip())

        if current_sender and current_content:
            results.append(
                schema.HumanMessage(
                    content="".join(current_content).strip(),
                    additional_kwargs={
                        "sender": current_sender,
                        "events": [{"message_time": current_timestamp}],
                    },
                )
            )

        return chat_loaders.ChatSession(messages=results)

    def lazy_load(self) -> Iterator[chat_loaders.ChatSession]:
        """
        Lazy load the messages from the chat file and yield them in the required format.

        Yields:
            A `ChatSession` object containing the loaded chat messages.
        """
        yield self._load_single_chat_session_from_txt(self.path)
