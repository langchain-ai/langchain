"""Loader that loads Obsidian directory dump."""
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class ObsidianLoader(BaseLoader):
    """Loader that loads Obsidian files from disk."""

    def __init__(self, path: str, encoding: str = "UTF-8"):
        """Initialize with path."""
        self.file_path = path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load documents."""
        ps = list(Path(self.file_path).glob("**/*.md"))
        docs = []
        for p in ps:
            with open(p, encoding=self.encoding) as f:
                text = f.read()
            metadata = {"source": str(p)}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
