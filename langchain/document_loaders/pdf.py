"""Loader that loads PDF files."""
from typing import Any, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredPDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load PDF files."""

    def _get_elements(self) -> List:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(filename=self.file_path)


class PDFMinerLoader(BaseLoader):
    """Loader that uses PDFMiner to load PDF files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            from pdfminer.high_level import extract_text  # noqa:F401
        except ImportError:
            raise ValueError(
                "pdfminer package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load file."""
        from pdfminer.high_level import extract_text

        text = extract_text(self.file_path)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]


class PyMuPDFLoader(BaseLoader):
    """Loader that uses PyMuPDF to load PDF files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import fitz  # noqa:F401
        except ImportError:
            raise ValueError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        self.file_path = file_path

    def load(self, **kwargs: Optional[Any]) -> List[Document]:
        """Load file."""
        import fitz

        doc = fitz.open(self.file_path)  # open document
        return [
            Document(
                page_content=page.get_text(**kwargs).encode("utf-8"),
                metadata={
                    "file_path": self.file_path,
                    "page_number": page.number + 1,
                    "total_pages": len(doc),
                }
                | doc.metadata,
            )
            for page in doc
        ]
