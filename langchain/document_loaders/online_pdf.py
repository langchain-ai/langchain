"""Loader that loads online PDF files."""

from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.pdf import BasePDFLoader, UnstructuredPDFLoader


class OnlinePDFLoader(BasePDFLoader):
    """Loader that loads online PDFs."""

    def load(self) -> List[Document]:
        """Load documents."""
        loader = UnstructuredPDFLoader(str(self.file_path))
        return loader.load()
