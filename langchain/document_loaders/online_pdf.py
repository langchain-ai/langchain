"""Loader that loads online PDF files."""

import tempfile
from typing import List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.pdf import UnstructuredPDFLoader


class OnlinePDFLoader(BaseLoader):
    """Loader that loads online PDFs."""

    def __init__(self, web_path: str):
        """Initialize with file path."""
        self.web_path = web_path

    def load(self) -> List[Document]:
        """Load documents."""
        r = requests.get(self.web_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/online_file.pdf"
            file = open(file_path, "wb")
            file.write(r.content)
            file.close()
            loader = UnstructuredPDFLoader(file_path)
            return loader.load()
