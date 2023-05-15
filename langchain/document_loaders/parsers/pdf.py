"""Module contains common parsers for PDFs."""
from typing import Any, Iterator, List, Mapping, Optional

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class PyPDFParser(BaseBlobParser):
    """Loads a PDF with pypdf and chunks at character level."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdf

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            yield from [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": blob.source, "page": page_number},
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]


class PDFMinerParser(BaseBlobParser):
    """Parse PDFs with PDFMiner."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        from pdfminer.high_level import extract_text

        with blob.as_bytes_io() as pdf_file_obj:
            text = extract_text(pdf_file_obj)
            metadata = {"source": blob.source}
            yield Document(page_content=text, metadata=metadata)


class PyMuPDFParser(BaseBlobParser):
    """Parse PDFs with PyMuPDF."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        self.text_kwargs = text_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz

        with blob.as_bytes_io() as file_path:
            doc = fitz.open(file_path)  # open document

            yield from [
                Document(
                    page_content=page.get_text(**self.text_kwargs),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.number,
                            "total_pages": len(doc),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc
            ]


class PyPDFium2Parser(BaseBlobParser):
    """Parse PDFs with PyPDFium2."""

    def __init__(self) -> None:
        """Initialize the parser."""
        try:
            import pypdfium2  # noqa:F401
        except ImportError:
            raise ValueError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdfium2

        # There's some finicky behavior with respect to segmentation faults
        # and closing file descriptors. For the time being, using bytes
        # and closing pdf reader manually.
        # May re-write at some point to use bytes io.
        pdf_reader = pypdfium2.PdfDocument(blob.as_bytes())
        docs: List[Document] = []
        try:
            for page_number, page in enumerate(pdf_reader):
                content = page.get_textpage().get_text_range()
                metadata = {"source": blob.source, "page": page_number}
                yield Document(page_content=str(content), metadata=metadata)
        finally:
            pdf_reader.close()
        yield from docs


class PDFPlumberParser(BaseBlobParser):
    """Parse PDFs with PDFPlumber."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
        """
        self.text_kwargs = text_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            yield from [
                Document(
                    page_content=page.extract_text(**self.text_kwargs),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.page_number,
                            "total_pages": len(doc.pages),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc.pages
            ]
