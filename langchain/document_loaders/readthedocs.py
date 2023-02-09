"""Loader that loads ReadTheDocs documentation directory dump."""
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class ReadTheDocsLoader(BaseLoader):
    """Loader that loads ReadTheDocs documentation directory dump."""

    def __init__(self, path: str):
        """Initialize path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        from bs4 import BeautifulSoup

        def _clean_data(data: str) -> str:
            soup = BeautifulSoup(data)
            text = soup.find_all("main", {"id": "main-content"})
            if len(text) != 0:
                text = text[0].get_text()
            else:
                text = ""
            return "\n".join([t for t in text.split("\n") if t])

        docs = []
        for p in Path(self.file_path).rglob("*"):
            if p.is_dir():
                continue
            with open(p) as f:
                text = _clean_data(f.read())
            metadata = {"source": str(p)}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
