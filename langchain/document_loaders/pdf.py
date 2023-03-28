"""Loader that loads PDF files."""
import os
import tempfile
from abc import ABC
from typing import Any, List, Optional
from urllib.parse import urlparse

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredPDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load PDF files."""

    def _get_elements(self) -> List:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(filename=self.file_path, **self.unstructured_kwargs)


class BasePDFLoader(BaseLoader, ABC):
    """Base loader class for PDF files.

    Defaults to check for local file, but if the file is a web path, it will download it
    to a temporary file, and use that, then clean up the temporary file after completion
    """

    file_path: str
    web_path: Optional[str] = None

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path, download it to a temporary file, and use that
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            r = requests.get(self.file_path)

            if r.status_code != 200:
                raise ValueError(
                    "Check the url of your file; returned status code %s"
                    % r.status_code
                )

            self.web_path = self.file_path
            self.temp_file = tempfile.NamedTemporaryFile()
            self.temp_file.write(r.content)
            self.file_path = self.temp_file.name
        elif not os.path.isfile(self.file_path):
            raise ValueError("File path %s is not a valid file or url" % self.file_path)

    def __del__(self) -> None:
        if hasattr(self, "temp_file"):
            self.temp_file.close()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)


class OnlinePDFLoader(BasePDFLoader):
    """Loader that loads online PDFs."""

    def load(self) -> List[Document]:
        """Load documents."""
        loader = UnstructuredPDFLoader(str(self.file_path))
        return loader.load()


class PyPDFLoader(BasePDFLoader):
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
        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        import pypdf

        with open(self.file_path, "rb") as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            return [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": self.file_path, "page": i},
                )
                for i, page in enumerate(pdf_reader.pages)
            ]


class PDFMinerLoader(BasePDFLoader):
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

        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load file."""
        from pdfminer.high_level import extract_text

        text = extract_text(self.file_path)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]


class PyMuPDFLoader(BasePDFLoader):
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

        super().__init__(file_path)

    def load(self, **kwargs: Optional[Any]) -> List[Document]:
        """Load file."""
        import fitz

        doc = fitz.open(self.file_path)  # open document
        file_path = self.file_path if self.web_path is None else self.web_path

        return [
            Document(
                page_content=page.get_text(**kwargs).encode("utf-8"),
                metadata=dict(
                    {
                        "source": file_path,
                        "file_path": file_path,
                        "page_number": page.number + 1,
                        "total_pages": len(doc),
                    },
                    **{
                        k: doc.metadata[k]
                        for k in doc.metadata
                        if type(doc.metadata[k]) in [str, int]
                    }
                ),
            )
            for page in doc
        ]
