import datetime
import json
from typing import List

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader


def concatenate_rows(message: dict, title: str) -> str:
    """
    Combine message information in a readable format ready to be used.
    Args:
        message: Message to be concatenated
        title: Title of the conversation

    Returns:
        Concatenated message
    """
    if not message:
        return ""

    sender = message["author"]["role"] if message["author"] else "unknown"
    text = message["content"]["parts"][0]
    date = datetime.datetime.fromtimestamp(message["create_time"]).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return f"{title} - {sender} on {date}: {text}\n\n"


class ChatGPTLoader(BaseLoader):
    """Load conversations from exported `ChatGPT` data."""

    def __init__(self, log_file: str, num_logs: int = -1):
        """Initialize a class object.

        Args:
            log_file: Path to the log file
            num_logs: Number of logs to load. If 0, load all logs.
        """
        self.log_file = log_file
        self.num_logs = num_logs

    def load(self) -> List[Document]:
        with open(self.log_file, encoding="utf8") as f:
            data = json.load(f)[: self.num_logs] if self.num_logs else json.load(f)

        documents = []
        for d in data:
            title = d["title"]
            messages = d["mapping"]
            text = "".join(
                [
                    concatenate_rows(messages[key]["message"], title)
                    for idx, key in enumerate(messages)
                    if not (
                        idx == 0
                        and messages[key]["message"]["author"]["role"] == "system"
                    )
                ]
            )
            metadata = {"source": str(self.log_file)}
            documents.append(Document(page_content=text, metadata=metadata))

        return documents
