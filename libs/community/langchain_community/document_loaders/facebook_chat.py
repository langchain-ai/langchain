import datetime
import json
from pathlib import Path
from typing import Iterator, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


def concatenate_rows(row: dict) -> str:
    """Combine message information in a readable format ready to be used.

    Args:
        row: dictionary containing message information.
    """
    sender = row["sender_name"]
    text = row["content"]
    date = datetime.datetime.fromtimestamp(row["timestamp_ms"] / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return f"{sender} on {date}: {text}\n\n"


class FacebookChatLoader(BaseLoader):
    """Load `Facebook Chat` messages directory dump."""

    def __init__(self, path: Union[str, Path]):
        """Initialize with a path."""
        self.file_path = path

    def lazy_load(self) -> Iterator[Document]:
        p = Path(self.file_path)

        with open(p, encoding="utf8") as f:
            d = json.load(f)

        text = "".join(
            concatenate_rows(message)
            for message in d["messages"]
            if message.get("content") and isinstance(message["content"], str)
        )
        metadata = {"source": str(p)}

        yield Document(page_content=text, metadata=metadata)
