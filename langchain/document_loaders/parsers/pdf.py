"""Implementations of PDF parsers.

This module provides various flavors of PDF parsers.

PDF parsers take a Blob and return a list of Document objects containing parsed content.
"""
from io import StringIO
from typing import Generator

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders.schema import Blob
from langchain.schema import Document


class PDFMinerParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Lazy parse PDF using PDFMiner."""
        try:
            from pdfminer.high_level import extract_text  # noqa:F401
        except ImportError:
            raise ValueError(
                "pdfminer package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        with blob.as_bytes_io() as pdf_file_obj:
            text = extract_text(pdf_file_obj)
        metadata = {"source": blob.source}
        yield [Document(page_content=text, metadata=metadata)]


class PyPDFLoader(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Load given path as pages."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ValueError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            for i, page in enumerate(pdf_reader.pages):
                yield Document(
                    page_content=page.extract_text(),
                    metadata={"source": blob.source, "page": i},
                )


class PDFMinerPDFasHTMLLoader(BaseBlobParser):
    """Loader that uses PDFMiner to load PDF files as HTML content."""

    def __init__(self, file_path: str):
        """Initialize with file path."""

        super().__init__(file_path)

    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Load file."""
        try:
            from pdfminer.high_level import extract_text_to_fp  # noqa:F401
        except ImportError:
            raise ValueError(
                "pdfminer package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output_string = StringIO()

        with blob.as_bytes_io() as pdf_file_obj:
            extract_text_to_fp(
                pdf_file_obj,  # type: ignore[arg-type]
                output_string,
                codec="",
                laparams=LAParams(),
                output_type="html",
            )
        metadata = {"source": blob.source}
        yield Document(page_content=output_string.getvalue(), metadata=metadata)


class PyMuPDFLoader(BaseBlobParser):
    """Loader that uses PyMuPDF to load PDF files."""

    def __init__(self, **kwargs) -> None:
        """Initialize with file path."""
        self.kwargs = kwargs

    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Load file."""
        try:
            import fitz  # noqa:F401
        except ImportError:
            raise ValueError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        with blob.as_bytes_io() as pdf_file_obj:
            doc = fitz.open(fileobj=pdf_file_obj)  # open document
            file_path = blob.source
            for page in doc:
                yield Document(
                    page_content=page.get_text(**self.kwargs).encode("utf-8"),
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
