"""Module contains common parsers for PDFs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document

if TYPE_CHECKING:
    import pdfplumber.page


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

    def __init__(
        self, text_kwargs: Optional[Mapping[str, Any]] = None, dedupe: bool = False
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            yield from [
                Document(
                    page_content=self._process_page_content(page),
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

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        return page.extract_text(**self.text_kwargs)


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


class DocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Forms Recognizer) and chunks at character level."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def _generate_docs(self, blob: Blob, result: Any) -> Iterator[Document]:
        page_content_dict = dict()

        for paragraph in result.paragraphs:
            page_number = paragraph.bounding_regions[0].page_number

            if page_number not in page_content_dict:
                page_content_dict[page_number] = ""

            page_content_dict[page_number] += paragraph.content + "\n\n"

        for page, content in page_content_dict.items():
            d = Document(
                page_content=content,
                metadata={
                    "source": blob.source,
                    "page": page,
                    "type": "TEXT",
                },
            )
            yield d

        if self.model in ["prebuilt-document", "prebuilt-layout", "prebuilt-invoice"]:
            for table_idx, table in enumerate(result.tables):
                page_num = table.bounding_regions[0].page_number
                headers = list()
                rows = dict()

                for cell in table.cells:
                    if cell.kind == "columnHeader":
                        headers.append(cell.content)
                    elif cell.kind == "content":
                        if cell.row_index not in rows:
                            rows[cell.row_index] = list()
                        rows[cell.row_index].append(cell.content)

                if headers:
                    hd = Document(
                        page_content=",".join(headers),
                        metadata={
                            "source": blob.source,
                            "page": page_num,
                            "type": "TABLE_HEADER",
                            "table_index": table_idx,
                        },
                    )
                    yield hd

                for _, row_cells in sorted(rows.items()):
                    rd = Document(
                        page_content=",".join(row_cells),
                        metadata={
                            "source": blob.source,
                            "page": page_num,
                            "type": "TABLE_ROW",
                            "table_index": table_idx,
                        },
                    )
                    yield rd

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(self.model, file_obj)
            result = poller.result()

            docs = self._generate_docs(blob, result)

            yield from docs
