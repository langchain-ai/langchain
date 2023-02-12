"""Load text files."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """Load text files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load from file path."""
        with open(self.file_path) as f:
            text = f.read()
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
