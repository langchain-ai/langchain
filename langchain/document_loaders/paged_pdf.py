"""Loads a PDF with pypdf and chunks at character level."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class PagedPDFSplitter(BaseLoader):
    """Loads a PDF with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ValueError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        self._file_path = file_path

    def load(self) -> List[Document]:
        """Load given path as pages."""
        import pypdf

        with open(self._file_path, "rb") as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            return [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": self._file_path, "page": i},
                )
                for i, page in enumerate(pdf_reader.pages)
            ]
