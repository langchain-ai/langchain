"""Class that helps load chat messages from a Discord chat.

To get the chat messages, just highlight them in the app,
right click, and select "Copy". Then paste in a text file.

"""
import logging
import re
from typing import Iterator, List, Optional

from langchain import schema
from langchain.chat_loaders import base as chat_loaders

logger = logging.getLogger(__name__)


class DiscordChatLoader(chat_loaders.BaseChatLoader):
    def __init__(self, path: str, user_name: Optional[str] = None):
        """
        Initialize the chat loader with the path to the exported Discord chat file.

        To make the export, open the Discord app, select a chat, highlight all messages
        you want to export, right click, and select "Copy". Then paste in a
        text file.

        :param path: Path to the exported Discord chat text file.
        :param user_name: Name of the user who will be mapped to the "AI" role.
        """
        self.path = path
        self.user_name = user_name
        self._message_line_regex = re.compile(
            r"(.+?) — (\w{3,9} \d{1,2}(?:st|nd|rd|th)?(?:, \d{4})? \d{1,2}:\d{2} (?:AM|PM)|Today at \d{1,2}:\d{2} (?:AM|PM)|Yesterday at \d{1,2}:\d{2} (?:AM|PM))",  # noqa
            flags=re.DOTALL,
        )

    def _load_single_chat_session_from_txt(
        self, file_path: str
    ) -> chat_loaders.ChatSession:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        results: List[schema.BaseMessage] = []
        current_sender = None
        current_timestamp = None
        current_content = []
        for line in lines:
            # Check if the line matches the pattern for the start of a message
            if re.match(
                r".+? — (\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} (?:AM|PM)|Today at \d{1,2}:\d{2} (?:AM|PM)|Yesterday at \d{1,2}:\d{2} (?:AM|PM))",  # noqa
                line,
            ):
                # If we were building a message, add it to the results
                if current_sender and current_content:
                    message_class = (
                        schema.AIMessage
                        if current_sender == self.user_name
                        else schema.HumanMessage
                    )
                    results.append(
                        message_class(
                            content="".join(current_content).strip(),
                            additional_kwargs={
                                "sender": current_sender,
                                "events": [{"message_time": current_timestamp}],
                            },
                        )
                    )
                # Start building the next message
                current_sender, current_timestamp = line.split(" — ")[:2]
                current_content = [
                    line[len(current_sender) + len(current_timestamp) + 4 :].strip()
                ]
            elif re.match(r"\[\d{1,2}:\d{2} (?:AM|PM)\]", line.strip()):
                message_class = (
                    schema.AIMessage
                    if current_sender == self.user_name
                    else schema.HumanMessage
                )
                results.append(
                    message_class(
                        content="".join(current_content).strip(),
                        additional_kwargs={
                            "sender": current_sender,
                            "events": [{"message_time": current_timestamp}],
                        },
                    )
                )
                current_timestamp = line.strip()[1:-1]

            else:
                current_content.append("\n" + line.strip())

        # Add the last message if there is one
        if current_sender and current_content:
            message = (
                schema.AIMessage
                if current_sender == self.user_name
                else schema.HumanMessage
            )
            results.append(
                message(
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

        :return: Iterator of dictionaries containing message role and content.
        """
        yield self._load_single_chat_session_from_txt(self.path)
