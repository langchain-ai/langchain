"""Module contains common parsers for PDFs."""
from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from urllib.parse import urlparse

import numpy as np

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document

if TYPE_CHECKING:
    import fitz.fitz
    import pdfminer.layout
    import pdfplumber.page
    import pypdf._page
    import pypdfium2._helpers.page


_PDF_FILTER_WITH_LOSS = ["DCTDecode", "DCT", "JPXDecode"]
_PDF_FILTER_WITHOUT_LOSS = [
    "LZWDecode",
    "LZW",
    "FlateDecode",
    "Fl",
    "ASCII85Decode",
    "A85",
    "ASCIIHexDecode",
    "AHx",
    "RunLengthDecode",
    "RL",
    "CCITTFaxDecode",
    "CCF",
    "JBIG2Decode",
]


def extract_from_images_with_rapidocr(
    images: Sequence[Union[Iterable[np.ndarray], bytes]]
) -> str:
    """Extract text from images with RapidOCR.

    Args:
        images: Images to extract text from.

    Returns:
        Text extracted from images.

    Raises:
        ImportError: If `rapidocr-onnxruntime` package is not installed.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise ImportError(
            "`rapidocr-onnxruntime` package not found, please install it with "
            "`pip install rapidocr-onnxruntime`"
        )
    ocr = RapidOCR()
    text = ""
    for img in images:
        result, _ = ocr(img)
        if result:
            result = [text[1] for text in result]
            text += "\n".join(result)
    return text


class PyPDFParser(BaseBlobParser):
    """Load `PDF` using `pypdf`"""

    def __init__(
        self, password: Optional[Union[str, bytes]] = None, extract_images: bool = False
    ):
        self.password = password
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdf

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
            yield from [
                Document(
                    page_content=page.extract_text()
                    + self._extract_images_from_page(page),
                    metadata={"source": blob.source, "page": page_number},
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]

    def _extract_images_from_page(self, page: pypdf._page.PageObject) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images or "/XObject" not in page["/Resources"].keys():
            return ""

        xObject = page["/Resources"]["/XObject"].get_object()  # type: ignore
        images = []
        for obj in xObject:
            if xObject[obj]["/Subtype"] == "/Image":
                if xObject[obj]["/Filter"][1:] in _PDF_FILTER_WITHOUT_LOSS:
                    height, width = xObject[obj]["/Height"], xObject[obj]["/Width"]

                    images.append(
                        np.frombuffer(xObject[obj].get_data(), dtype=np.uint8).reshape(
                            height, width, -1
                        )
                    )
                elif xObject[obj]["/Filter"][1:] in _PDF_FILTER_WITH_LOSS:
                    images.append(xObject[obj].get_data())
                else:
                    warnings.warn("Unknown PDF Filter!")
        return extract_from_images_with_rapidocr(images)


class PDFMinerParser(BaseBlobParser):
    """Parse `PDF` using `PDFMiner`."""

    def __init__(self, extract_images: bool = False):
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        if not self.extract_images:
            from pdfminer.high_level import extract_text

            with blob.as_bytes_io() as pdf_file_obj:
                text = extract_text(pdf_file_obj)
                metadata = {"source": blob.source}
                yield Document(page_content=text, metadata=metadata)
        else:
            import io

            from pdfminer.converter import PDFPageAggregator, TextConverter
            from pdfminer.layout import LAParams
            from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
            from pdfminer.pdfpage import PDFPage

            text_io = io.StringIO()
            with blob.as_bytes_io() as pdf_file_obj:
                pages = PDFPage.get_pages(pdf_file_obj)
                rsrcmgr = PDFResourceManager()
                device_for_text = TextConverter(rsrcmgr, text_io, laparams=LAParams())
                device_for_image = PDFPageAggregator(rsrcmgr, laparams=LAParams())
                interpreter_for_text = PDFPageInterpreter(rsrcmgr, device_for_text)
                interpreter_for_image = PDFPageInterpreter(rsrcmgr, device_for_image)
                for i, page in enumerate(pages):
                    interpreter_for_text.process_page(page)
                    interpreter_for_image.process_page(page)
                    content = text_io.getvalue() + self._extract_images_from_page(
                        device_for_image.get_result()
                    )
                    text_io.truncate(0)
                    text_io.seek(0)
                    metadata = {"source": blob.source, "page": str(i)}
                    yield Document(page_content=content, metadata=metadata)

    def _extract_images_from_page(self, page: pdfminer.layout.LTPage) -> str:
        """Extract images from page and get the text with RapidOCR."""
        import pdfminer

        def get_image(layout_object: Any) -> Any:
            if isinstance(layout_object, pdfminer.layout.LTImage):
                return layout_object
            if isinstance(layout_object, pdfminer.layout.LTContainer):
                for child in layout_object:
                    return get_image(child)
            else:
                return None

        images = []

        for img in list(filter(bool, map(get_image, page))):
            if img.stream["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    np.frombuffer(img.stream.get_data(), dtype=np.uint8).reshape(
                        img.stream["Height"], img.stream["Width"], -1
                    )
                )
            elif img.stream["Filter"].name in _PDF_FILTER_WITH_LOSS:
                images.append(img.stream.get_data())
            else:
                warnings.warn("Unknown PDF Filter!")
        return extract_from_images_with_rapidocr(images)


class PyMuPDFParser(BaseBlobParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(
        self,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        extract_images: bool = False,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        self.text_kwargs = text_kwargs or {}
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz

        with blob.as_bytes_io() as file_path:
            doc = fitz.open(file_path)  # open document

            yield from [
                Document(
                    page_content=page.get_text(**self.text_kwargs)
                    + self._extract_images_from_page(doc, page),
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

    def _extract_images_from_page(
        self, doc: fitz.fitz.Document, page: fitz.fitz.Page
    ) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""
        import fitz

        img_list = page.get_images()
        imgs = []
        for img in img_list:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            imgs.append(
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, -1
                )
            )
        return extract_from_images_with_rapidocr(imgs)


class PyPDFium2Parser(BaseBlobParser):
    """Parse `PDF` with `PyPDFium2`."""

    def __init__(self, extract_images: bool = False) -> None:
        """Initialize the parser."""
        try:
            import pypdfium2  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )
        self.extract_images = extract_images

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
                    content += "\n" + self._extract_images_from_page(page)
                    page.close()
                    metadata = {"source": blob.source, "page": page_number}
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()

    def _extract_images_from_page(self, page: pypdfium2._helpers.page.PdfPage) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""

        import pypdfium2.raw as pdfium_c

        images = list(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)))

        images = list(map(lambda x: x.get_bitmap().to_numpy(), images))
        return extract_from_images_with_rapidocr(images)


class PDFPlumberParser(BaseBlobParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(
        self,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        dedupe: bool = False,
        extract_images: bool = False,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            yield from [
                Document(
                    page_content=self._process_page_content(page)
                    + "\n"
                    + self._extract_images_from_page(page),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.page_number - 1,
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

    def _extract_images_from_page(self, page: pdfplumber.page.Page) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""

        images = []
        for img in page.images:
            if img["stream"]["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    np.frombuffer(img["stream"].get_data(), dtype=np.uint8).reshape(
                        img["stream"]["Height"], img["stream"]["Width"], -1
                    )
                )
            elif img["stream"]["Filter"].name in _PDF_FILTER_WITH_LOSS:
                images.append(img["stream"].get_data())
            else:
                warnings.warn("Unknown PDF Filter!")

        return extract_from_images_with_rapidocr(images)


class AmazonTextractPDFParser(BaseBlobParser):
    """Send `PDF` files to `Amazon Textract` and parse them.

    For parsing multi-page PDFs, they have to reside on S3.

    The AmazonTextractPDFLoader calls the
    [Amazon Textract Service](https://aws.amazon.com/textract/)
    to convert PDFs into a Document structure.
    Single and multi-page documents are supported with up to 3000 pages
    and 512 MB of size.

    For the call to be successful an AWS account is required,
    similar to the
    [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
    requirements.

    Besides the AWS configuration, it is very similar to the other PDF
    loaders, while also supporting JPEG, PNG and TIFF and non-native
    PDF formats.

    ```python
    from langchain.document_loaders import AmazonTextractPDFLoader
    loader=AmazonTextractPDFLoader("example_data/alejandro_rosalez_sample-small.jpeg")
    documents = loader.load()
    ```

    One feature is the linearization of the output.
    When using the features LAYOUT, FORMS or TABLES together with Textract

    ```python
    from langchain.document_loaders import AmazonTextractPDFLoader
    # you can mix and match each of the features
    loader=AmazonTextractPDFLoader(
        "example_data/alejandro_rosalez_sample-small.jpeg",
        textract_features=["TABLES", "LAYOUT"])
    documents = loader.load()
    ```

    it will generate output that formats the text in reading order and
    try to output the information in a tabular structure or
    output the key/value pairs with a colon (key: value).
    This helps most LLMs to achieve better accuracy when
    processing these texts.

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
            import textractor.entities.document as textractor

            self.tc = tc
            self.textractor = textractor

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
        has to be set to the S3 URI and for single page docs
        the blob.data is taken
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

        document = self.textractor.Document.open(textract_response_json)

        linearizer_config = self.textractor.TextLinearizationConfig(
            hide_figure_layout=True,
            title_prefix="# ",
            section_header_prefix="## ",
            list_element_prefix="*",
        )
        for idx, page in enumerate(document.pages):
            yield Document(
                page_content=page.get_text(config=linearizer_config),
                metadata={"source": blob.source, "page": idx + 1},
            )


class DocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Forms Recognizer) and chunks at character level."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def _generate_docs(self, blob: Blob, result: Any) -> Iterator[Document]:
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "source": blob.source,
                    "page": p.page_number,
                },
            )
            yield d

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(self.model, file_obj)
            result = poller.result()

            docs = self._generate_docs(blob, result)

            yield from docs
