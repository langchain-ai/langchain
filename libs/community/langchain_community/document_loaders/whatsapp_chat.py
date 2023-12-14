import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


def concatenate_rows(date: str, sender: str, text: str) -> str:
    """Combine message information in a readable format ready to be used."""
    return f"{sender} on {date}: {text}\n\n"


class WhatsAppChatLoader(BaseLoader):
    """Load `WhatsApp` messages text file."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.file_path)
        text_content = ""

        with open(p, encoding="utf8") as f:
            lines = f.readlines()

        message_line_regex = r"""
            \[?
            (
                \d{1,4}
                [\/.]
                \d{1,2}
                [\/.]
                \d{1,4}
                ,\s
                \d{1,2}
                :\d{2}
                (?:
                    :\d{2}
                )?
                (?:[\s_](?:AM|PM))?
            )
            \]?
            [\s-]*
            ([~\w\s]+)
            [:]+
            \s
            (.+)
        """
        ignore_lines = ["This message was deleted", "<Media omitted>"]
        for line in lines:
            result = re.match(
                message_line_regex, line.strip(), flags=re.VERBOSE | re.IGNORECASE
            )
            if result:
                date, sender, text = result.groups()
                if text not in ignore_lines:
                    text_content += concatenate_rows(date, sender, text)

        metadata = {"source": str(p)}

        return [Document(page_content=text_content, metadata=metadata)]
