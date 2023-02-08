"""Loader that loads .txt web files."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class GutenbergLoader(BaseLoader):
    """Loader that uses unstructured to load HTML files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        assert file_path.startswith("https://www.gutenberg.org"), "file path must start with 'https://www.gutenberg.org'"
        assert file_path.endswith(".txt"), "file path must end with '.txt'"
        
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load file."""
        from urllib.request import urlopen

        elements = urlopen(self.file_path)
        text = "\n\n".join([str(el.decode("utf-8-sig")) for el in elements])
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
