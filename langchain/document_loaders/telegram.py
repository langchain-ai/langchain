"""Loader that loads Telegram chat json dump."""
import json
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def concatenate_rows(row: dict) -> str:
    """Combine message information in a readable format ready to be used."""
    date = row["date"]
    sender = row["from"]
    text = row["text"]
    return f"{sender} on {date}: {text}\n\n"


class TelegramChatLoader(BaseLoader):
    """Loader that loads Telegram chat json directory dump."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.file_path)

        with open(p, encoding="utf8") as f:
            d = json.load(f)

        text = "".join(
            concatenate_rows(message)
            for message in d["messages"]
            if message["type"] == "message" and isinstance(message["text"], str)
        )
        metadata = {"source": str(p)}

        return [Document(page_content=text, metadata=metadata)]
