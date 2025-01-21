"""Module contains common parsers for PDFs."""

from __future__ import annotations

import html
import io
import logging
import threading
import warnings
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from urllib.parse import urlparse

import numpy
import numpy as np
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.images import (
    BaseImageBlobParser,
    RapidOCRBlobParser,
)

if TYPE_CHECKING:
    import pdfminer
    import pdfplumber
    import pymupdf
    import pypdf
    import pypdfium2
    from textractor.data.text_linearization_config import TextLinearizationConfig

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
    images: Sequence[Union[Iterable[np.ndarray], bytes]],
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


logger = logging.getLogger(__name__)

_FORMAT_IMAGE_STR = "\n\n{image_text}\n\n"
_JOIN_IMAGES = "\n"
_JOIN_TABLES = "\n"
_DEFAULT_PAGES_DELIMITER = "\n\f"

_STD_METADATA_KEYS = {"source", "total_pages", "creationdate", "creator", "producer"}


def _format_inner_image(blob: Blob, content: str, format: str) -> str:
    """Format the content of the image with the source of the blob.

    blob: The blob containing the image.
    format::
      The format for the parsed output.
      - "text" = return the content as is
      - "markdown-img" = wrap the content into an image markdown link, w/ link
      pointing to (`![body)(#)`]
      - "html-img" = wrap the content as the `alt` text of an tag and link to
      (`<img alt="{body}" src="#"/>`)
    """
    if content:
        source = blob.source or "#"
        if format == "markdown-img":
            content = content.replace("]", r"\\]")
            content = f"![{content}]({source})"
        elif format == "html-img":
            content = (
                f'<img alt="{html.escape(content, quote=True)} ' f'src="{source}" />'
            )
    return content


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate that the metadata has all the standard keys and the page is an integer.

    The standard keys are:
    - source
    - total_page
    - creationdate
    - creator
    - producer

    Validate that page is an integer if it is present.
    """
    if not _STD_METADATA_KEYS.issubset(metadata.keys()):
        raise ValueError("The PDF parser must valorize the standard metadata.")
    if not isinstance(metadata.get("page", 0), int):
        raise ValueError("The PDF metadata page must be a integer.")
    return metadata


def _purge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Purge metadata from unwanted keys and normalize key names.

    Args:
        metadata: The original metadata dictionary.

    Returns:
        The cleaned and normalized the key format of metadata dictionary.
    """
    new_metadata: dict[str, Any] = {}
    map_key = {
        "page_count": "total_pages",
        "file_path": "source",
    }
    for k, v in metadata.items():
        if type(v) not in [str, int]:
            v = str(v)
        if k.startswith("/"):
            k = k[1:]
        k = k.lower()
        if k in ["creationdate", "moddate"]:
            try:
                new_metadata[k] = datetime.strptime(
                    v.replace("'", ""), "D:%Y%m%d%H%M%S%z"
                ).isoformat("T")
            except ValueError:
                new_metadata[k] = v
        elif k in map_key:
            # Normalize key with others PDF parser
            new_metadata[map_key[k]] = v
            new_metadata[k] = v
        elif isinstance(v, str):
            new_metadata[k] = v.strip()
        elif isinstance(v, int):
            new_metadata[k] = v
    return new_metadata


_PARAGRAPH_DELIMITER = [
    "\n\n\n",
    "\n\n",
]  # To insert images or table in the middle of the page.


def _merge_text_and_extras(extras: list[str], text_from_page: str) -> str:
    """Insert extras such as image/table in a text between two paragraphs if possible,
    else at the end of the text.

    Args:
        extras: List of extra content (images/tables) to insert.
        text_from_page: The text content from the page.

    Returns:
        The merged text with extras inserted.
    """

    def _recurs_merge_text_and_extras(
        extras: list[str], text_from_page: str, recurs: bool
    ) -> Optional[str]:
        if extras:
            for delim in _PARAGRAPH_DELIMITER:
                pos = text_from_page.rfind(delim)
                if pos != -1:
                    # search penultimate, to bypass an error in footer
                    previous_text = None
                    if recurs:
                        previous_text = _recurs_merge_text_and_extras(
                            extras, text_from_page[:pos], False
                        )
                    if previous_text:
                        all_text = previous_text + text_from_page[pos:]
                    else:
                        all_extras = ""
                        str_extras = "\n\n".join(filter(lambda x: x, extras))
                        if str_extras:
                            all_extras = delim + str_extras
                        all_text = (
                            text_from_page[:pos] + all_extras + text_from_page[pos:]
                        )
                    break
            else:
                all_text = None
        else:
            all_text = text_from_page
        return all_text

    all_text = _recurs_merge_text_and_extras(extras, text_from_page, True)
    if not all_text:
        all_extras = ""
        str_extras = "\n\n".join(filter(lambda x: x, extras))
        if str_extras:
            all_extras = _PARAGRAPH_DELIMITER[-1] + str_extras
        all_text = text_from_page + all_extras

    return all_text


class PyPDFParser(BaseBlobParser):
    """Load `PDF` using `pypdf`"""

    def __init__(
        self,
        password: Optional[Union[str, bytes]] = None,
        extract_images: bool = False,
        *,
        extraction_mode: str = "plain",
        extraction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.password = password
        self.extract_images = extract_images
        self.extraction_mode = extraction_mode
        self.extraction_kwargs = extraction_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "`pypdf` package not found, please install it with "
                "`pip install pypdf`"
            )

        def _extract_text_from_page(page: pypdf.PageObject) -> str:
            """Extract text from image given the version of pypdf."""
            if pypdf.__version__.startswith("3"):
                return page.extract_text()
            else:
                return page.extract_text(
                    extraction_mode=self.extraction_mode,  # type: ignore[arg-type]
                    **self.extraction_kwargs,  # type: ignore[arg-type]
                )

        with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)

            yield from [
                Document(
                    page_content=_extract_text_from_page(page=page)
                    + self._extract_images_from_page(page),
                    metadata={
                        "source": blob.source,
                        "page": page_number,
                        "page_label": pdf_reader.page_labels[page_number],
                    },
                    # type: ignore[attr-defined]
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]

    def _extract_images_from_page(self, page: pypdf.PageObject) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images or "/XObject" not in page["/Resources"].keys():  # type: ignore[attr-defined]
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

    def __init__(self, extract_images: bool = False, *, concatenate_pages: bool = True):
        """Initialize a parser based on PDFMiner.

        Args:
            extract_images: Whether to extract images from PDF.
            concatenate_pages: If True, concatenate all PDF pages into one a single
                               document. Otherwise, return one document per page.
        """
        self.extract_images = extract_images
        self.concatenate_pages = concatenate_pages

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""

        if not self.extract_images:
            try:
                from pdfminer.high_level import extract_text
            except ImportError:
                raise ImportError(
                    "`pdfminer` package not found, please install it with "
                    "`pip install pdfminer.six`"
                )

            with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
                if self.concatenate_pages:
                    text = extract_text(pdf_file_obj)
                    metadata = {"source": blob.source}  # type: ignore[attr-defined]
                    yield Document(page_content=text, metadata=metadata)
                else:
                    from pdfminer.pdfpage import PDFPage

                    pages = PDFPage.get_pages(pdf_file_obj)
                    for i, _ in enumerate(pages):
                        text = extract_text(pdf_file_obj, page_numbers=[i])
                        metadata = {"source": blob.source, "page": str(i)}  # type: ignore[attr-defined]
                        yield Document(page_content=text, metadata=metadata)
        else:
            import io

            from pdfminer.converter import PDFPageAggregator, TextConverter
            from pdfminer.layout import LAParams
            from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
            from pdfminer.pdfpage import PDFPage

            text_io = io.StringIO()
            with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
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
                    metadata = {"source": blob.source, "page": str(i)}  # type: ignore[attr-defined]
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

        for img in filter(bool, map(get_image, page)):
            img_filter = img.stream["Filter"]
            if isinstance(img_filter, list):
                filter_names = [f.name for f in img_filter]
            else:
                filter_names = [img_filter.name]

            without_loss = any(
                name in _PDF_FILTER_WITHOUT_LOSS for name in filter_names
            )
            with_loss = any(name in _PDF_FILTER_WITH_LOSS for name in filter_names)
            non_matching = {name for name in filter_names} - {
                *_PDF_FILTER_WITHOUT_LOSS,
                *_PDF_FILTER_WITH_LOSS,
            }

            if without_loss and with_loss:
                warnings.warn(
                    "Image has both lossy and lossless filters. Defaulting to lossless"
                )

            if non_matching:
                warnings.warn(f"Unknown PDF Filter(s): {non_matching}")

            if without_loss:
                images.append(
                    np.frombuffer(img.stream.get_data(), dtype=np.uint8).reshape(
                        img.stream["Height"], img.stream["Width"], -1
                    )
                )
            elif with_loss:
                images.append(img.stream.get_data())

        return extract_from_images_with_rapidocr(images)


class PyMuPDFParser(BaseBlobParser):
    """Parse a blob from a PDF using `PyMuPDF` library.

    This class provides methods to parse a blob from a PDF document, supporting various
    configurations such as handling password-protected PDFs, extracting images, and
    defining extraction mode.
    It integrates the 'PyMuPDF' library for PDF processing and offers synchronous blob
    parsing.

    Examples:
        Setup:

        .. code-block:: bash

            pip install -U langchain-community pymupdf

        Load a blob from a PDF file:

        .. code-block:: python

            from langchain_core.documents.base import Blob

            blob = Blob.from_path("./example_data/layout-parser-paper.pdf")

        Instantiate the parser:

        .. code-block:: python

            from langchain_community.document_loaders.parsers import PyMuPDFParser

            parser = PyMuPDFParser(
                # password = None,
                mode = "single",
                pages_delimiter = "\n\f",
                # extract_images = True,
                # images_parser = TesseractBlobParser(),
                # extract_tables="markdown",
                # extract_tables_settings=None,
                # text_kwargs=None,
            )

        Lazily parse the blob:

        .. code-block:: python

            docs = []
            docs_lazy = parser.lazy_parse(blob)

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)
    """

    # PyMuPDF is not thread safe.
    # See https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
    _lock = threading.Lock()

    def __init__(
        self,
        text_kwargs: Optional[dict[str, Any]] = None,
        extract_images: bool = False,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "page",
        pages_delimiter: str = _DEFAULT_PAGES_DELIMITER,
        images_parser: Optional[BaseImageBlobParser] = None,
        images_inner_format: Literal["text", "markdown-img", "html-img"] = "text",
        extract_tables: Union[Literal["csv", "markdown", "html"], None] = None,
        extract_tables_settings: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a parser based on PyMuPDF.

        Args:
            password: Optional password for opening encrypted PDFs.
            mode: The extraction mode, either "single" for the entire document or "page"
                for page-wise extraction.
            pages_delimiter: A string delimiter to separate pages in single-mode
                extraction.
            extract_images: Whether to extract images from the PDF.
            images_parser: Optional image blob parser.
            images_inner_format: The format for the parsed output.
                - "text" = return the content as is
                - "markdown-img" = wrap the content into an image markdown link, w/ link
                pointing to (`![body)(#)`]
                - "html-img" = wrap the content as the `alt` text of an tag and link to
                (`<img alt="{body}" src="#"/>`)
            extract_tables: Whether to extract tables in a specific format, such as
                "csv", "markdown", or "html".
            extract_tables_settings: Optional dictionary of settings for customizing
                table extraction.

        Returns:
            This method does not directly return data. Use the `parse` or `lazy_parse`
            methods to retrieve parsed documents with content and metadata.

        Raises:
            ValueError: If the mode is not "single" or "page".
            ValueError: If the extract_tables format is not "markdown", "html",
            or "csv".
        """
        super().__init__()
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        if extract_tables and extract_tables not in ["markdown", "html", "csv"]:
            raise ValueError("mode must be markdown")

        self.mode = mode
        self.pages_delimiter = pages_delimiter
        self.password = password
        self.text_kwargs = text_kwargs or {}
        if extract_images and not images_parser:
            images_parser = RapidOCRBlobParser()
        self.extract_images = extract_images
        self.images_inner_format = images_inner_format
        self.images_parser = images_parser
        self.extract_tables = extract_tables
        self.extract_tables_settings = extract_tables_settings

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        return self._lazy_parse(
            blob,
        )

    def _lazy_parse(
        self,
        blob: Blob,
        # text-kwargs is present for backwards compatibility.
        # Users should not use it directly.
        text_kwargs: Optional[dict[str, Any]] = None,
    ) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob.
        Insert image, if possible, between two paragraphs.
        In this way, a paragraph can be continued on the next page.

        Args:
            blob: The blob to parse.
            text_kwargs: Optional keyword arguments to pass to the `get_text` method.
                If provided at run time, it will override the default text_kwargs.

        Raises:
            ImportError: If the `pypdf` package is not found.

        Yield:
            An iterator over the parsed documents.
        """
        try:
            import pymupdf

            text_kwargs = text_kwargs or self.text_kwargs
            if not self.extract_tables_settings:
                from pymupdf.table import (
                    DEFAULT_JOIN_TOLERANCE,
                    DEFAULT_MIN_WORDS_HORIZONTAL,
                    DEFAULT_MIN_WORDS_VERTICAL,
                    DEFAULT_SNAP_TOLERANCE,
                )

                self.extract_tables_settings = {
                    # See https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables
                    "clip": None,
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "vertical_lines": None,
                    "horizontal_lines": None,
                    "snap_tolerance": DEFAULT_SNAP_TOLERANCE,
                    "snap_x_tolerance": None,
                    "snap_y_tolerance": None,
                    "join_tolerance": DEFAULT_JOIN_TOLERANCE,
                    "join_x_tolerance": None,
                    "join_y_tolerance": None,
                    "edge_min_length": 3,
                    "min_words_vertical": DEFAULT_MIN_WORDS_VERTICAL,
                    "min_words_horizontal": DEFAULT_MIN_WORDS_HORIZONTAL,
                    "intersection_tolerance": 3,
                    "intersection_x_tolerance": None,
                    "intersection_y_tolerance": None,
                    "text_tolerance": 3,
                    "text_x_tolerance": 3,
                    "text_y_tolerance": 3,
                    "strategy": None,  # offer abbreviation
                    "add_lines": None,  # optional user-specified lines
                }
        except ImportError:
            raise ImportError(
                "pymupdf package not found, please install it "
                "with `pip install pymupdf`"
            )

        with PyMuPDFParser._lock:
            with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
                if blob.data is None:  # type: ignore[attr-defined]
                    doc = pymupdf.open(file_path)
                else:
                    doc = pymupdf.open(stream=file_path, filetype="pdf")
                if doc.is_encrypted:
                    doc.authenticate(self.password)
                doc_metadata = self._extract_metadata(doc, blob)
                full_content = []
                for page in doc:
                    all_text = self._get_page_content(doc, page, text_kwargs).strip()
                    if self.mode == "page":
                        yield Document(
                            page_content=all_text,
                            metadata=_validate_metadata(
                                doc_metadata | {"page": page.number}
                            ),
                        )
                    else:
                        full_content.append(all_text)

                if self.mode == "single":
                    yield Document(
                        page_content=self.pages_delimiter.join(full_content),
                        metadata=_validate_metadata(doc_metadata),
                    )

    def _get_page_content(
        self,
        doc: pymupdf.Document,
        page: pymupdf.Page,
        text_kwargs: dict[str, Any],
    ) -> str:
        """Get the text of the page using PyMuPDF and RapidOCR and issue a warning
        if it is empty.

        Args:
            doc: The PyMuPDF document object.
            page: The PyMuPDF page object.
            blob: The blob being parsed.

        Returns:
            str: The text content of the page.
        """
        text_from_page = page.get_text(**{**self.text_kwargs, **text_kwargs})
        images_from_page = self._extract_images_from_page(doc, page)
        tables_from_page = self._extract_tables_from_page(page)
        extras = []
        if images_from_page:
            extras.append(images_from_page)
        if tables_from_page:
            extras.append(tables_from_page)
        all_text = _merge_text_and_extras(extras, text_from_page)

        return all_text

    def _extract_metadata(self, doc: pymupdf.Document, blob: Blob) -> dict:
        """Extract metadata from the document and page.

        Args:
            doc: The PyMuPDF document object.
            blob: The blob being parsed.

        Returns:
            dict: The extracted metadata.
        """
        return _purge_metadata(
            dict(
                {
                    "producer": "PyMuPDF",
                    "creator": "PyMuPDF",
                    "creationdate": "",
                    "source": blob.source,  # type: ignore[attr-defined]
                    "file_path": blob.source,  # type: ignore[attr-defined]
                    "total_pages": len(doc),
                },
                **{
                    k: doc.metadata[k]
                    for k in doc.metadata
                    if isinstance(doc.metadata[k], (str, int))
                },
            )
        )

    def _extract_images_from_page(
        self, doc: pymupdf.Document, page: pymupdf.Page
    ) -> str:
        """Extract images from a PDF page and get the text using images_to_text.

        Args:
            doc: The PyMuPDF document object.
            page: The PyMuPDF page object.

        Returns:
            str: The extracted text from the images on the page.
        """
        if not self.images_parser:
            return ""
        import pymupdf

        img_list = page.get_images()
        images = []
        for img in img_list:
            if self.images_parser:
                xref = img[0]
                pix = pymupdf.Pixmap(doc, xref)
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, -1
                )
                image_bytes = io.BytesIO()
                numpy.save(image_bytes, image)
                blob = Blob.from_data(
                    image_bytes.getvalue(), mime_type="application/x-npy"
                )
                image_text = next(self.images_parser.lazy_parse(blob)).page_content

                images.append(
                    _format_inner_image(blob, image_text, self.images_inner_format)
                )
        return _FORMAT_IMAGE_STR.format(
            image_text=_JOIN_IMAGES.join(filter(None, images))
        )

    def _extract_tables_from_page(self, page: pymupdf.Page) -> str:
        """Extract tables from a PDF page.

        Args:
            page: The PyMuPDF page object.

        Returns:
            str: The extracted tables in the specified format.
        """
        if self.extract_tables is None:
            return ""
        import pymupdf

        tables_list = list(
            pymupdf.table.find_tables(page, **self.extract_tables_settings)
        )
        if tables_list:
            if self.extract_tables == "markdown":
                return _JOIN_TABLES.join([table.to_markdown() for table in tables_list])
            elif self.extract_tables == "html":
                return _JOIN_TABLES.join(
                    [
                        table.to_pandas().to_html(
                            header=False,
                            index=False,
                            bold_rows=False,
                        )
                        for table in tables_list
                    ]
                )
            elif self.extract_tables == "csv":
                return _JOIN_TABLES.join(
                    [
                        table.to_pandas().to_csv(
                            header=False,
                            index=False,
                        )
                        for table in tables_list
                    ]
                )
            else:
                raise ValueError(
                    f"extract_tables {self.extract_tables} not implemented"
                )
        return ""


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

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pypdfium2

        # pypdfium2 is really finicky with respect to closing things,
        # if done incorrectly creates seg faults.
        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    content += "\n" + self._extract_images_from_page(page)
                    page.close()
                    metadata = {"source": blob.source, "page": page_number}  # type: ignore[attr-defined]
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
        try:
            import PIL  # noqa:F401
        except ImportError:
            raise ImportError(
                "pillow package not found, please install it with"
                " `pip install pillow`"
            )
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            doc = pdfplumber.open(file_path)  # open document

            yield from [
                Document(
                    page_content=self._process_page_content(page)
                    + "\n"
                    + self._extract_images_from_page(page),
                    metadata=dict(
                        {
                            "source": blob.source,  # type: ignore[attr-defined]
                            "file_path": blob.source,  # type: ignore[attr-defined]
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
        from PIL import Image

        if not self.extract_images:
            return ""

        images = []
        for img in page.images:
            if img["stream"]["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                if img["stream"]["BitsPerComponent"] == 1:
                    images.append(
                        np.array(
                            Image.frombytes(
                                "1",
                                (img["stream"]["Width"], img["stream"]["Height"]),
                                img["stream"].get_data(),
                            ).convert("L")
                        )
                    )
                else:
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
    from langchain_community.document_loaders import AmazonTextractPDFLoader
    loader=AmazonTextractPDFLoader("example_data/alejandro_rosalez_sample-small.jpeg")
    documents = loader.load()
    ```

    One feature is the linearization of the output.
    When using the features LAYOUT, FORMS or TABLES together with Textract

    ```python
    from langchain_community.document_loaders import AmazonTextractPDFLoader
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
        *,
        linearization_config: Optional[TextLinearizationConfig] = None,
    ) -> None:
        """Initializes the parser.

        Args:
            textract_features: Features to be used for extraction, each feature
                               should be passed as an int that conforms to the enum
                               `Textract_Features`, see `amazon-textract-caller` pkg
            client: boto3 textract client
            linearization_config: Config to be used for linearization of the output
                                  should be an instance of TextLinearizationConfig from
                                  the `textractor` pkg
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

            if linearization_config is not None:
                self.linearization_config = linearization_config
            else:
                self.linearization_config = self.textractor.TextLinearizationConfig(
                    hide_figure_layout=True,
                    title_prefix="# ",
                    section_header_prefix="## ",
                    list_element_prefix="*",
                )
        except ImportError:
            raise ImportError(
                "Could not import amazon-textract-caller or "
                "amazon-textract-textractor python package. Please install it "
                "with `pip install amazon-textract-caller` & "
                "`pip install amazon-textract-textractor`."
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

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Iterates over the Blob pages and returns an Iterator with a Document
        for each page, like the other parsers If multi-page document, blob.path
        has to be set to the S3 URI and for single page docs
        the blob.data is taken
        """

        url_parse_result = urlparse(str(blob.path)) if blob.path else None  # type: ignore[attr-defined]
        # Either call with S3 path (multi-page) or with bytes (single-page)
        if (
            url_parse_result
            and url_parse_result.scheme == "s3"
            and url_parse_result.netloc
        ):
            textract_response_json = self.tc.call_textract(
                input_document=str(blob.path),  # type: ignore[attr-defined]
                features=self.textract_features,
                boto3_textract_client=self.boto3_textract_client,
            )
        else:
            textract_response_json = self.tc.call_textract(
                input_document=blob.as_bytes(),  # type: ignore[attr-defined]
                features=self.textract_features,
                call_mode=self.tc.Textract_Call_Mode.FORCE_SYNC,
                boto3_textract_client=self.boto3_textract_client,
            )

        document = self.textractor.Document.open(textract_response_json)

        for idx, page in enumerate(document.pages):
            yield Document(
                page_content=page.get_text(config=self.linearization_config),
                metadata={"source": blob.source, "page": idx + 1},  # type: ignore[attr-defined]
            )


class DocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Form Recognizer) and chunks at character level."""

    def __init__(self, client: Any, model: str):
        warnings.warn(
            "langchain_community.document_loaders.parsers.pdf.DocumentIntelligenceParser"
            "and langchain_community.document_loaders.pdf.DocumentIntelligenceLoader"
            " are deprecated. Please upgrade to "
            "langchain_community.document_loaders.DocumentIntelligenceLoader "
            "for any file parsing purpose using Azure Document Intelligence "
            "service."
        )
        self.client = client
        self.model = model

    def _generate_docs(self, blob: Blob, result: Any) -> Iterator[Document]:  # type: ignore[valid-type]
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "source": blob.source,  # type: ignore[attr-defined]
                    "page": p.page_number,
                },
            )
            yield d

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:  # type: ignore[attr-defined]
            poller = self.client.begin_analyze_document(self.model, file_obj)
            result = poller.result()

            docs = self._generate_docs(blob, result)

            yield from docs
