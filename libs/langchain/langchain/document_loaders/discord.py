from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    import pandas as pd


class DiscordChatLoader(BaseLoader):
    """Load `Discord` chat logs."""

    def __init__(self, chat_log: pd.DataFrame, user_id_col: str = "ID"):
        """Initialize with a Pandas DataFrame containing chat logs.

        Args:
            chat_log: Pandas DataFrame containing chat logs.
            user_id_col: Name of the column containing the user ID. Defaults to "ID".
        """
        if not isinstance(chat_log, pd.DataFrame):
            raise ValueError(
                f"Expected chat_log to be a pd.DataFrame, got {type(chat_log)}"
            )
        self.chat_log = chat_log
        self.user_id_col = user_id_col

    def load(self) -> List[Document]:
        """Load all chat messages."""
        result = []
        for _, row in self.chat_log.iterrows():
            user_id = row[self.user_id_col]
            metadata = row.to_dict()
            metadata.pop(self.user_id_col)
            result.append(Document(page_content=user_id, metadata=metadata))
        return result
