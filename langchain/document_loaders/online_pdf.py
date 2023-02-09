"""Loader that loads online PDF files."""
import os
from typing import List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders.base import BaseLoader


class OnlinePDFLoader(BaseLoader):
    """Loader that loads online PDFs."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load file."""
        r = requests.get(self.file_path)
        with open("online_file.pdf", "wb") as f:
            f.write(r.content)

        loader = UnstructuredPDFLoader(f.name)
        data = loader.load()
        data[0].metadata = {"source": self.file_path}

        f.close()
        os.remove(f.name)

        return data
