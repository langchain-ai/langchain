"""Module contains common parsers for PDFs."""
from typing import Any, Iterator, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class PyPDFParser(BaseBlobParser):
    """Load `PDF` using `pypdf` and chunk at character level."""

    def __init__(self, password: Optional[Union[str, bytes]] = None):
        self.password = password

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdf

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
            yield from [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": blob.source, "page": page_number},
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]


class PDFMinerParser(BaseBlobParser):
    """Parse `PDF` using `PDFMiner`."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        from pdfminer.high_level import extract_text

        with blob.as_bytes_io() as pdf_file_obj:
            text = extract_text(pdf_file_obj)
            metadata = {"source": blob.source}
            yield Document(page_content=text, metadata=metadata)


class PyMuPDFParser(BaseBlobParser):
    """Parse `PDF` using `PyMuPDF`."""

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
    """Parse `PDF` with `PyPDFium2`."""

    def __init__(self) -> None:
        """Initialize the parser."""
        try:
            import pypdfium2  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdfium2

        # pypdfium2 is really finicky with respect to closing things,
        # if done incorrectly creates seg faults.
        with blob.as_bytes_io() as file_path:
            pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    page.close()
                    metadata = {"source": blob.source, "page": page_number}
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()


class PDFPlumberParser(BaseBlobParser):
    """Parse `PDF` with `PDFPlumber`."""

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


class AmazonTextractPDFParser(BaseBlobParser):
    """Send `PDF` files to `Amazon Textract` and parse them.

    For parsing multi-page PDFs, they have to reside on S3.
    """

    def __init__(
        self,
        textract_features: Optional[Sequence[int]] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initializes the parser.

        Args:
            textract_features: Features to be used for extraction, each feature
                               should be passed as an int that conforms to the enum
                               `Textract_Features`, see `amazon-textract-caller` pkg
            client: boto3 textract client
        """

        try:
            import textractcaller as tc

            self.tc = tc
            if textract_features is not None:
                self.textract_features = [
                    tc.Textract_Features(f) for f in textract_features
                ]
            else:
                self.textract_features = []
        except ImportError:
            raise ImportError(
                "Could not import amazon-textract-caller python package. "
                "Please install it with `pip install amazon-textract-caller`."
            )

        if not client:
            try:
                import boto3

                self.boto3_textract_client = boto3.client("textract")
            except ImportError:
                raise ImportError(
                    "Could not import boto3 python package. "
                    "Please install it with `pip install boto3`."
                )
        else:
            self.boto3_textract_client = client

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Iterates over the Blob pages and returns an Iterator with a Document
        for each page, like the other parsers If multi-page document, blob.path
        has to be set to the S3 URI and for single page docs the blob.data is taken
        """

        url_parse_result = urlparse(str(blob.path)) if blob.path else None
        # Either call with S3 path (multi-page) or with bytes (single-page)
        if (
            url_parse_result
            and url_parse_result.scheme == "s3"
            and url_parse_result.netloc
        ):
            textract_response_json = self.tc.call_textract(
                input_document=str(blob.path),
                features=self.textract_features,
                boto3_textract_client=self.boto3_textract_client,
            )
        else:
            textract_response_json = self.tc.call_textract(
                input_document=blob.as_bytes(),
                features=self.textract_features,
                call_mode=self.tc.Textract_Call_Mode.FORCE_SYNC,
                boto3_textract_client=self.boto3_textract_client,
            )

        current_text = ""
        current_page = 1
        for block in textract_response_json["Blocks"]:
            if "Page" in block and not (int(block["Page"]) == current_page):
                yield Document(
                    page_content=current_text,
                    metadata={"source": blob.source, "page": current_page},
                )
                current_text = ""
                current_page = int(block["Page"])
            if "Text" in block:
                current_text += block["Text"] + " "

        yield Document(
            page_content=current_text,
            metadata={"source": blob.source, "page": current_page},
        )
