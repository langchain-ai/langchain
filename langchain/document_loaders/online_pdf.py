"""Loader that loads online PDF files."""

import requests

from langchain.document_loaders import UnstructuredPDFLoader


class OnlinePDFLoader(UnstructuredPDFLoader):
    """Loader that loads online PDFs."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        r = requests.get(file_path)
        with open("example_data/online_file.pdf", "wb") as f:
            f.write(r.content)
        self.file_path = "example_data/online_file.pdf"
