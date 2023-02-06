"""Loading logic for loading documents from a directory."""
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class DirectoryLoader(BaseLoader):
    """Loading logic for loading documents from a directory."""

    def __init__(self, path: str, glob: str = "**/*"):
        """Initialize with path to directory and how to glob over it."""
        self.path = path
        self.glob = glob

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.path)
        docs = []
        for i in p.glob(self.glob):
            if i.is_file():
                sub_docs = UnstructuredFileLoader(str(i)).load()
                docs.extend(sub_docs)
        return docs
