"""Module contains common parsers for PDFs."""

import base64
import html
import io
import logging
import threading
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)
from urllib.parse import urlparse

import numpy as np
from langchain_core._api.deprecation import (
    deprecated,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob

if TYPE_CHECKING:
    import pdfplumber
    import pdfplumber.page
    import pymupdf
    import pymupdf.pymupdf
    import pypdf
    import pypdf._page
    import pypdfium2
    import pypdfium2._helpers.page
    from pypdf import PageObject
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

logger = logging.getLogger(__name__)

_format_image_str = "\n\n{image_text}\n\n"
_join_images = "\n"
_join_tables = "\n"
_default_page_delimitor = "\n\f"


def purge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Purge metadata from unwanted keys and normalize key names.

    Args:
        metadata (dict[str, Any]): The original metadata dictionary.

    Returns:
        dict[str, Any]: The cleaned and normalized metadata dictionary.
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
            # Normliaze key with others PDF parser
            new_metadata[map_key[k]] = v
            new_metadata[k] = v
        elif isinstance(v, str):
            new_metadata[k] = v.strip()
        elif isinstance(v, int):
            new_metadata[k] = v
    return new_metadata


_delim = ["\n\n\n", "\n\n"]  # To insert images or table in the middle of the page.


def __merge_text_and_extras(
    extras: list[str], text_from_page: str, recurs: bool
) -> Optional[str]:
    """
    Insert extras such as image/table in a text between two paragraphs if possible.

    Args:
        extras (list[str]): List of extra content (images/tables) to insert.
        text_from_page (str): The text content from the page.
        recurs (bool): Flag to indicate if the function should recurse.

    Returns:
        Optional[str]: The merged text with extras inserted, or None if no insertion
        point is found.
    """
    if extras:
        for delim in _delim:
            pos = text_from_page.rfind(delim)
            if pos != -1:
                # search penultimate, to bypass an error in footer
                previous_text = None
                if recurs:
                    previous_text = __merge_text_and_extras(
                        extras, text_from_page[:pos], False
                    )
                if previous_text:
                    all_text = previous_text + text_from_page[pos:]
                else:
                    all_extras = ""
                    str_extras = "\n\n".join(filter(lambda x: x, extras))
                    if str_extras:
                        all_extras = delim + str_extras
                    all_text = text_from_page[:pos] + all_extras + text_from_page[pos:]
                break
        else:
            all_text = None
    else:
        all_text = text_from_page
    return all_text


def _merge_text_and_extras(extras: list[str], text_from_page: str) -> str:
    """
    Insert extras such as image/table in a text between two paragraphs if possible,
    else at the end of the text.

    Args:
        extras (list[str]): List of extra content (images/tables) to insert.
        text_from_page (str): The text content from the page.

    Returns:
        str: The merged text with extras inserted.
    """
    all_text = __merge_text_and_extras(extras, text_from_page, True)
    if not all_text:
        all_extras = ""
        str_extras = "\n\n".join(filter(lambda x: x, extras))
        if str_extras:
            all_extras = _delim[-1] + str_extras
        all_text = text_from_page + all_extras

    return all_text


@deprecated(since="3.0.0", alternative="Use Parser.images_to_text()")
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


# Type to change the function to convert images to text.
CONVERT_IMAGE_TO_TEXT = Optional[Callable[[Iterable[np.ndarray]], Iterator[str]]]


def convert_images_to_text_with_rapidocr(
    # Default to text format to be compatible with previous versions.
    *,
    format: Literal["text", "markdown", "html"] = "text",
) -> CONVERT_IMAGE_TO_TEXT:
    """
    Return a function to convert images to text using RapidOCR.

    Note: RapidOCR is compatible english and chinese languages.

    Args:
        format: Format of the output text. Either "text" or "markdown".
    """

    def _convert_images_to_text(images: Iterable[np.ndarray]) -> Iterator[str]:
        """Extract text from images.
        Can be overloaded to use another OCR algorithm, or to use
        a multimodal model to describe the images.

        Args:
            images: Images to extract text from.

        Yield:
            Text extracted from each image.

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

        for img in images:
            ocr_result, _ = ocr(img)
            if ocr_result:
                result = ("\n".join([text[1] for text in ocr_result])).strip()
                if result:
                    if format == "markdown":
                        result = result.replace("]", r"\\]")
                        result = f"![{result}](.)"
                    elif format == "html":
                        result = f'<img alt="{html.escape(result, quote=True)}" />'
                logger.debug("RapidOCR text: %s", result.replace("\n", "\\n"))
                yield result
            else:
                yield ""

    return _convert_images_to_text


def convert_images_to_text_with_tesseract(
    # Default to text format to be compatible with previous versions.
    *,
    format: Literal["text", "markdown", "html"] = "text",
    langs: list[str] = ["eng"],
) -> CONVERT_IMAGE_TO_TEXT:
    """
    Return a function to convert images to text using RapidOCR.
    Note: RapidOCR is compatible english and chinese languages.
    Args:
        format: Format of the output text. Either "text" or "markdown".
    """

    def _convert_images_to_text(images: Iterable[np.ndarray]) -> Iterator[str]:
        """Extract text from images.
        Can be overloaded to use another OCR algorithm, or to use
        a multimodal model to describe the images.

        Args:
            images: Images to extract text from.

        Yield:
            Text extracted from each image.

        Raises:
            ImportError: If `rapidocr-onnxruntime` package is not installed.
        """
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "`pytesseract` package not found, please install it with "
                "`pip install pytesseract`"
            )

        for img in images:
            result = pytesseract.image_to_string(img, lang="+".join(langs)).strip()
            if result:
                if format == "markdown":
                    result = result.replace("]", r"\\]")
                    result = f"![{result}](.)"
                elif format == "html":
                    result = f'<img alt="{html.escape(result, quote=True)}" />'
            logger.debug("Tesseract text: %s", result.replace("\n", "\\n"))
            yield result

    return _convert_images_to_text


_prompt_images_to_description = PromptTemplate.from_template(
    """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval \
    and extract all the text from the image."""
)


def convert_images_to_description(
    model: BaseChatModel,
    *,
    prompt: BasePromptTemplate = _prompt_images_to_description,
    format: Literal["text", "markdown", "html"] = "markdown",
) -> CONVERT_IMAGE_TO_TEXT:
    """
    Return a function to convert images to text using a multimodal model.

    Args:
        model: Multimodal model to use to describe the images.
        prompt: Optional prompt to use to describe the images.
        format: Format of the output text. Either "text" or "markdown".


    Returns:
        A function to extract text from images using the multimodal model.
    """

    def _convert_images_to_description(
        images: Iterable[np.ndarray],
    ) -> Iterator[str]:
        """Describe an image and extract text.
        Use a multimodal model to describe the images.

        Args:
            images: Images to extract text from.

        Yield:
            Text extracted from each image.

        Raises:
            ImportError: If `rapidocr-onnxruntime` package is not installed.
        """

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "`PIL` package not found, please install it with `pip install pillow`"
            )
        chat = model
        for image in images:  # TODO: Add a batch processing?
            image_bytes = io.BytesIO()
            Image.fromarray(image).save(image_bytes, format="PNG")
            img_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            msg = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt.format()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                },
                            },
                        ]
                    )
                ]
            )
            result = msg.content
            assert isinstance(result, str)
            if result:
                if format == "markdown":
                    result = result.replace("]", r"\\]")
                    result = f"![{result}](.)"
                elif format == "html":
                    result = f'<img alt="{str(html.escape(result, quote=True))}" />'
                elif format == "text":
                    pass
                else:
                    raise ValueError(f"Unknown format: {format}")
            logger.debug("LLM description: %s", result.replace("\n", "\\n"))
            yield result

    return _convert_images_to_description


class ImagesPdfParser(BaseBlobParser):
    """Abstract interface for blob parsers with OCR."""

    def __init__(
        self,
        extract_images: bool,
        images_to_text: CONVERT_IMAGE_TO_TEXT,
    ):
        """Extract text from images.

        Args:
            extract_images: Whether to extract images from PDF.
            images_to_text: Function to extract text from images.
        """
        self.extract_images = extract_images

        self.convert_image_to_text = cast(
            Callable[[Iterable[np.ndarray]], Iterator[str]],
            (images_to_text or convert_images_to_text_with_rapidocr()),
        )


class PyPDFParser(ImagesPdfParser):
    """Parse a blob from a PDF using `pypdf` library.

    This class provides methods to parse a blob from a PDF document, supporting various
    configurations such as handling password-protected PDFs, extracting images, and
    defining extraction mode.
    It integrates the 'pypdf' library for PDF processing and offers synchronous blob
    parsing.

    Examples:
        Setup:

        .. code-block:: bash

            pip install -U langchain-core langchain-community pypdf

        Load a blob from a PDF file:

        .. code-block:: python

            from langchain_core.documents.base import Blob

            blob = Blob.from_path("./example_data/layout-parser-paper.pdf")

        Instantiate the parser:

        .. code-block:: python

            from langchain_community.document_loaders.parsers import PyPDFParser

            parser = PyPDFParser(
                # password = None,
                # extract_images = False,
                mode = "single",
                pages_delimitor = "\n\f",
                # images_to_text = None,
                # extraction_mode = "plain",
                # extraction_kwargs = None,
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

    def __init__(
        self,
        password: Optional[Union[str, bytes]] = None,
        extract_images: bool = False,
        *,  # Move on top ?
        mode: Literal["single", "page"] = "page",
        pages_delimitor: str = _default_page_delimitor,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        extraction_mode: Literal["plain", "layout"] = "plain",
        extraction_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initialize a parser based on PyPDF.

        Args:
            password: Optional password for opening encrypted PDFs.
            extract_images: Whether to extract images from the PDF.
            mode: The extraction mode, either "single" for the entire document or "page"
                for page-wise extraction.
            pages_delimitor: A string delimiter to separate pages in single-mode
                extraction.
            images_to_text: Function or callable to convert images to text during
                extraction.
            extraction_mode: “plain” for legacy functionality, “layout” for experimental
                layout mode functionality
            extraction_kwargs: Optional additional parameters for the extraction
                process.

        Returns:
            This method does not directly return data. Use the `parse` or `lazy_parse`
            methods to retrieve parsed documents with content and metadata.

        Raises:
            ImportError: If the `pypdf` package is not installed.
            ValueError: If the `mode` is not "single" or "page".
        """
        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        self.password = password
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.extraction_mode = extraction_mode
        self.extraction_kwargs = extraction_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """
        Lazily parse the blob.
        Insert image, if possible, between two paragraphs.
        In this way, a paragraph can be continued on the next page.

        Args:
            blob: The blob to parse.

        Yield:
            An iterator over the parsed documents.
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )

        def _extract_text_from_page(page: "PageObject") -> str:
            """
            Extract text from image given the version of pypdf.

            Args:
                page: The page object to extract text from.

            Returns:
                str: The extracted text.
            """

            def before(
                operator: Any,
                operand_arguments: Any,
                current_transformation_matrix: Any,
                text_matrix: Any,
            ) -> None:
                pass

            def after(
                operator: Any,
                operand_arguments: Any,
                current_transformation_matrix: Any,
                text_matrix: Any,
            ) -> None:
                pass

            def text(
                text: Any,
                current_transformation_matrix: Any,
                text_matrix: Any,
                font_dictionary: Any,
                font_size: Any,
            ) -> Any:
                pass

            if pypdf.__version__.startswith("3"):
                return page.extract_text()
            else:
                return page.extract_text(
                    extraction_mode=self.extraction_mode,
                    **self.extraction_kwargs,
                    visitor_operand_before=before,
                    visitor_operand_after=after,
                    visitor_text=text,
                )

        with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)

            doc_metadata = purge_metadata(
                cast(dict, pdf_reader.metadata or {})
                | {
                    "source": blob.source,
                    "total_pages": len(pdf_reader.pages),
                }
            )
            single_texts = []
            for page_number, page in enumerate(pdf_reader.pages):
                text_from_page = _extract_text_from_page(page=page)
                images_from_page = self.extract_images_from_page(page)
                all_text = _merge_text_and_extras(
                    [images_from_page], text_from_page
                ).strip()
                if self.mode == "page":
                    yield Document(
                        page_content=all_text,
                        metadata=doc_metadata | {"page": page_number},
                    )
                else:
                    single_texts.append(all_text)
            if self.mode == "single":
                yield Document(
                    page_content=self.pages_delimitor.join(single_texts),
                    metadata=doc_metadata,
                )

    def extract_images_from_page(self, page: "pypdf._page.PageObject") -> str:
        """Extract images from a PDF page and get the text using RapidOCR.

        Args:
            page: The page object from which to extract images.

        Returns:
            str: The extracted text from the images on the page.
        """
        from PIL import Image

        if (
            not self.extract_images
            or "/XObject" not in cast(dict, page["/Resources"]).keys()
        ):
            return ""

        xObject = page["/Resources"]["/XObject"].get_object()  # type: ignore[index]
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
                    images.append(
                        np.array(Image.open(io.BytesIO(xObject[obj].get_data())))
                    )

                else:
                    logger.warning("Unknown PDF Filter!")
        return _format_image_str.format(
            image_text=_join_images.join(self.convert_image_to_text(images))
        )


class PDFMinerParser(ImagesPdfParser):
    """Parse a blob from a PDF using `pdfminer.six` library.

    This class provides methods to parse a blob from a PDF document, supporting various
    configurations such as handling password-protected PDFs, extracting images, and
    defining extraction mode.
    It integrates the 'pdfminer.six' library for PDF processing and offers synchronous
    blob parsing.

    Examples:
        Setup:

        .. code-block:: bash

            pip install -U langchain-core langchain-community pdfminer.six

        Load a blob from a PDF file:

        .. code-block:: python

            from langchain_core.documents.base import Blob

            blob = Blob.from_path("./example_data/layout-parser-paper.pdf")

        Instantiate the parser:

        .. code-block:: python

            from langchain_community.document_loaders.parsers import PDFMinerParser

            parser = PDFMinerParser(
                # extract_images = False,
                # password = None,
                mode = "single",
                pages_delimitor = "\n\f",
                # images_to_text = None,
                # concatenate_pages = None,
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

    def __init__(
        self,
        extract_images: bool = False,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        concatenate_pages: Optional[bool] = None,
    ):
        """Initialize a parser based on PDFMiner.

        Args:
            extract_images: Whether to extract images from PDF.
            password: Optional password for opening encrypted PDFs.
            mode: Extraction mode to use. Either "single" or "page" for page-wise
                extraction.
            pages_delimitor: A string delimiter to separate pages in single-mode
                extraction.
            images_to_text: Function or callable to convert images to text during
                extraction.
            concatenate_pages: Deprecated. If True, concatenate all PDF pages
                into one a single document. Otherwise, return one document per page.

        Returns:
            This method does not directly return data. Use the `parse` or `lazy_parse`
            methods to retrieve parsed documents with content and metadata.

        Raises:
            ImportError: If the `pdfminer.six` package is not installed.
            ValueError: If the `mode` is not "single" or "page".

        Warnings:
            `concatenate_pages` parameter is deprecated. Use `mode='single' or 'page'
            instead.
        """
        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.password = password
        self.extract_images = extract_images
        self.images_to_text = images_to_text
        if concatenate_pages is not None:
            logger.warning(
                "`concatenate_pages` parameter is deprecated. "
                "Use `mode='single' or 'page'` instead."
            )
            self.mode = "single" if concatenate_pages else "page"

    @staticmethod
    def decode_text(s: Union[bytes, str]) -> str:
        """
        Decodes a PDFDocEncoding string to Unicode.
        Adds py3 compatibility to pdfminer's version.

        Args:
            s: The string to decode.

        Returns:
            str: The decoded Unicode string.
        """
        from pdfminer.utils import PDFDocEncoding

        if isinstance(s, bytes) and s.startswith(b"\xfe\xff"):
            return str(s[2:], "utf-16be", "ignore")
        try:
            ords = (ord(c) if isinstance(c, str) else c for c in s)
            return "".join(PDFDocEncoding[o] for o in ords)
        except IndexError:
            return str(s)

    @staticmethod
    def resolve_and_decode(obj: Any) -> Any:
        """
        Recursively resolve the metadata values.

        Args:
            obj: The object to resolve and decode. It can be of any type.

        Returns:
            The resolved and decoded object.
        """
        from pdfminer.psparser import PSLiteral

        if hasattr(obj, "resolve"):
            obj = obj.resolve()
        if isinstance(obj, list):
            return list(map(PDFMinerParser.resolve_and_decode, obj))
        elif isinstance(obj, PSLiteral):
            return PDFMinerParser.decode_text(obj.name)
        elif isinstance(obj, (str, bytes)):
            return PDFMinerParser.decode_text(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = PDFMinerParser.resolve_and_decode(v)
            return obj

        return obj

    def _get_metadata(
        self,
        fp: BinaryIO,
        password: str = "",
        caching: bool = True,
    ) -> dict[str, Any]:
        """
        Extract metadata from a PDF file.

        Args:
            fp: The file pointer to the PDF file.
            password: The password for the PDF file, if encrypted. Defaults to an empty
                string.
            caching: Whether to cache the PDF structure. Defaults to True.

        Returns:
            Metadata of the PDF file.
        """
        from pdfminer.pdfpage import PDFDocument, PDFPage, PDFParser

        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        doc = PDFDocument(parser, password=password, caching=caching)
        metadata = {}

        for info in doc.info:
            metadata.update(info)
        for k, v in metadata.items():
            try:
                metadata[k] = PDFMinerParser.resolve_and_decode(v)
            except Exception as e:  # pragma: nocover
                # This metadata value could not be parsed. Instead of failing the PDF
                # read, treat it as a warning only if `strict_metadata=False`.
                logger.warning(
                    '[WARNING] Metadata key "%s" could not be parsed due to '
                    "exception: %s",
                    k,
                    str(e),
                )

        # Count number of pages.
        metadata["total_pages"] = len(list(PDFPage.create_pages(doc)))

        return metadata

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """
        Lazily parse the blob.

        Args:
            blob: The blob to parse.

        Yield:
            An iterator over the parsed documents.
        """
        try:
            from pdfminer.converter import PDFLayoutAnalyzer
            from pdfminer.layout import (
                LAParams,
                LTContainer,
                LTImage,
                LTItem,
                LTPage,
                LTText,
                LTTextBox,
            )
            from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
            from pdfminer.pdfpage import PDFPage
        except ImportError:
            raise ImportError(
                "pdfminer package not found, please install it "
                "with `pip install pdfminer`"
            )
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pdfminer package not found, please install it "
                "with `pip install pillow`"
            )

        with blob.as_bytes_io() as pdf_file_obj, TemporaryDirectory() as tempdir:
            pages = PDFPage.get_pages(pdf_file_obj, password=self.password or "")
            rsrcmgr = PDFResourceManager()
            doc_metadata = purge_metadata(
                self._get_metadata(pdf_file_obj, password=self.password or "")
            )
            doc_metadata["source"] = blob.source

            class Visitor(PDFLayoutAnalyzer):
                def __init__(
                    self,
                    rsrcmgr: PDFResourceManager,
                    pageno: int = 1,
                    laparams: Optional[LAParams] = None,
                ) -> None:
                    super().__init__(rsrcmgr, pageno=pageno, laparams=laparams)

                def receive_layout(me, ltpage: LTPage) -> None:
                    def render(item: LTItem) -> None:
                        if isinstance(item, LTContainer):
                            for child in item:
                                render(child)
                        elif isinstance(item, LTText):
                            text_io.write(item.get_text())
                        if isinstance(item, LTTextBox):
                            text_io.write("\n")
                        elif isinstance(item, LTImage):
                            if self.extract_images and self.images_to_text:
                                from pdfminer.image import ImageWriter

                                image_writer = ImageWriter(tempdir)
                                filename = image_writer.export_image(item)
                                img = np.array(Image.open(Path(tempdir) / filename))
                                image_text = next(self.images_to_text([img]))
                                if image_text:
                                    text_io.write(
                                        _format_image_str.format(image_text=image_text)
                                    )
                        else:
                            pass

                    render(ltpage)

            text_io = io.StringIO()
            visitor_for_all = PDFPageInterpreter(
                rsrcmgr, Visitor(rsrcmgr, laparams=LAParams())
            )
            all_content = []
            for i, page in enumerate(pages):
                text_io.truncate(0)
                text_io.seek(0)
                visitor_for_all.process_page(page)

                all_text = text_io.getvalue()
                # For legacy compatibility, net strip()
                all_text = all_text.strip()
                if self.mode == "page":
                    text_io.truncate(0)
                    text_io.seek(0)
                    yield Document(
                        page_content=all_text, metadata=doc_metadata | {"page": i}
                    )
                else:
                    if all_text.endswith("\f"):
                        all_text = all_text[:-1]
                    all_content.append(all_text)
            if self.mode == "single":
                # Add page_delimitor between pages
                document_content = self.pages_delimitor.join(all_content)
                yield Document(
                    page_content=document_content,
                    metadata=doc_metadata,
                )


class PyMuPDFParser(ImagesPdfParser):
    """Parse a blob from a PDF using `PyMuPDF` library.

    This class provides methods to parse a blob from a PDF document, supporting various
    configurations such as handling password-protected PDFs, extracting images, and
    defining extraction mode.
    It integrates the 'PyMuPDF' library for PDF processing and offers synchronous blob
    parsing.

    Examples:
        Setup:

        .. code-block:: bash

            pip install -U langchain-core langchain-community pymupdf

        Load a blob from a PDF file:

        .. code-block:: python

            from langchain_core.documents.base import Blob

            blob = Blob.from_path("./example_data/layout-parser-paper.pdf")

        Instantiate the parser:

        .. code-block:: python

            from langchain_community.document_loaders.parsers import PyMuPDFParser

            parser = PyMuPDFParser(
                # password=None,
                mode="page",
                pages_delimitor="\n\f",
                # extract_images=False,
                # images_to_text=None,
                # extract_tables=None,
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
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "page",
        pages_delimitor: str = _default_page_delimitor,
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        extract_tables: Union[Literal["csv", "markdown", "html"], None] = None,
        extract_tables_settings: Optional[dict[str, Any]] = None,
        text_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a parser based on PyMuPDF.

        Args:
            password: Optional password for opening encrypted PDFs.
            mode: Mode of parsing, either "single" for a single document or "page" for
                individual pages.
            pages_delimitor: Delimiter to use between pages when mode is "single".
            extract_images: Whether to extract images from the PDF.
            images_to_text:  Function or callable to convert images to text during
                extraction.
            extract_tables: Whether to extract tables in a specific format, such as
                "csv", "markdown", or "html".
            extract_tables_settings: Optional dictionary of settings for customizing
                table extraction.
            **kwargs: Additional keyword arguments for customizing text extraction
                behavior.

        Returns:
            This method does not directly return data. Use the `parse` or `lazy_parse`
            methods to retrieve parsed documents with content and metadata.

        Raises:
            ImportError: If the `pymupdf` package is not installed.
            ValueError: If the mode is not "single" or "page".
            ValueError: If the extract_tables format is not "markdown", "html",
            or "csv".
        """
        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        if extract_tables and extract_tables not in ["markdown", "html", "csv"]:
            raise ValueError("mode must be markdown")

        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.password = password  # PPR: https://github.com/pymupdf/RAG/pull/170
        self.text_kwargs = text_kwargs or {}
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_tables_settings = extract_tables_settings

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """
        Lazily parse the blob.

        Args:
            blob: The blob to parse.

        Yield:
            An iterator over the parsed documents.
        """
        try:
            import pymupdf

            if not self.extract_tables_settings:
                from pymupdf.table import (
                    DEFAULT_JOIN_TOLERANCE,
                    DEFAULT_MIN_WORDS_HORIZONTAL,
                    DEFAULT_MIN_WORDS_VERTICAL,
                    DEFAULT_SNAP_TOLERANCE,
                )

                self.extract_tables_settings = {
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
                    all_text = self._get_page_content(doc, page, blob).strip()
                    if self.mode == "page":
                        yield Document(
                            page_content=all_text,
                            metadata=(doc_metadata | {"page": page.number}),
                        )
                    else:
                        full_content.append(all_text)

                if self.mode == "single":
                    yield Document(
                        page_content=self.pages_delimitor.join(full_content),
                        metadata=doc_metadata,
                    )

    def _get_page_content(
        self, doc: "pymupdf.pymupdf.Document", page: "pymupdf.pymupdf.Page", blob: Blob
    ) -> str:
        """
        Get the text of the page using PyMuPDF and RapidOCR and issue a warning
        if it is empty.

        Args:
            doc: The PyMuPDF document object.
            page: The PyMuPDF page object.
            blob: The blob being parsed.

        Returns:
            str: The text content of the page.
        """
        text_from_page = page.get_text(**self.text_kwargs)
        images_from_page = self._extract_images_from_page(doc, page)
        tables_from_page = self._extract_tables_from_page(page)
        extras = []
        if images_from_page:
            extras.append(images_from_page)
        if tables_from_page:
            extras.append(tables_from_page)
        all_text = _merge_text_and_extras(extras, text_from_page)

        if not all_text:
            logger.warning(
                "Warning: Empty content on page %s of document %s",
                page.number,
                blob.source,
            )

        return all_text

    def _extract_metadata(self, doc: "pymupdf.pymupdf.Document", blob: Blob) -> dict:
        """Extract metadata from the document and page.

        Args:
            doc: The PyMuPDF document object.
            blob: The blob being parsed.

        Returns:
            dict: The extracted metadata.
        """
        return purge_metadata(
            dict(
                {
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
        self, doc: "pymupdf.pymupdf.Document", page: "pymupdf.pymupdf.Page"
    ) -> str:
        """Extract images from a PDF page and get the text using RapidOCR.

        Args:
            doc: The PyMuPDF document object.
            page: The PyMuPDF page object.

        Returns:
            str: The extracted text from the images on the page.
        """
        if not self.extract_images:
            return ""
        import pymupdf

        img_list = page.get_images()
        images = []
        for img in img_list:
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            images.append(
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, -1
                )
            )
            _format_image_str.format(
                image_text=_join_images.join(self.convert_image_to_text(images))
            )

        return _format_image_str.format(
            image_text=_join_images.join(self.convert_image_to_text(images))
        )

    def _extract_tables_from_page(self, page: "pymupdf.pymupdf.Page") -> str:
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
                return _join_tables.join([table.to_markdown() for table in tables_list])
            elif self.extract_tables == "html":
                return _join_tables.join(
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
                return _join_tables.join(
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


class PyPDFium2Parser(ImagesPdfParser):
    """Parse a blob from a PDF using `PyPDFium2` library.

    This class provides methods to parse a blob from a PDF document, supporting various
    configurations such as handling password-protected PDFs, extracting images, and
    defining extraction mode.
    It integrates the 'PyPDFium2' library for PDF processing and offers synchronous
    blob parsing.

    Examples:
        Setup:

        .. code-block:: bash

            pip install -U langchain-core langchain-community pypdfium2

        Load a blob from a PDF file:

        .. code-block:: python

            from langchain_core.documents.base import Blob

            blob = Blob.from_path("./example_data/layout-parser-paper.pdf")

        Instantiate the parser:

        .. code-block:: python

            from langchain_community.document_loaders.parsers import PyPDFium2Parser

            parser = PyPDFium2Parser(
                # password=None,
                mode="page",
                pages_delimitor="\n\f",
                # extract_images=False,
                # images_to_text=None,
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

    # PyPDFium2 is not thread safe.
    # See https://pypdfium2.readthedocs.io/en/stable/python_api.html#thread-incompatibility
    _lock = threading.Lock()

    def __init__(
        self,
        extract_images: bool = False,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "page",
        pages_delimitor: str = _default_page_delimitor,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
    ) -> None:
        """Initialize a parser based on PyPDFium2.

        Args:
            password: Optional password for opening encrypted PDFs.
            mode: Mode of parsing, either "single" for a single document or "page" for
                individual pages.
            pages_delimitor: Delimiter to use between pages when mode is "single".
            extract_images: Whether to extract images from the PDF.
            images_to_text:  Function or callable to convert images to text during
                extraction.

        Returns:
            This method does not directly return data. Use the `parse` or `lazy_parse`
            methods to retrieve parsed documents with content and metadata.

        Raises:
            ImportError: If the `pypdfium2` package is not installed.
            ValueError: If the mode is not "single" or "page".
        """
        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.password = password

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob.

        Args:
            blob: The blob to parse.

        Yields:
            An iterator over the parsed documents.
        """
        try:
            import pypdfium2
        except ImportError:
            raise ImportError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )

        # pypdfium2 is really finicky with respect to closing things,
        # if done incorrectly creates seg faults.
        with PyPDFium2Parser._lock:
            with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
                pdf_reader = None
                try:
                    pdf_reader = pypdfium2.PdfDocument(
                        file_path, password=self.password, autoclose=True
                    )
                    full_content = []

                    doc_metadata = purge_metadata(pdf_reader.get_metadata_dict())
                    doc_metadata["source"] = blob.source
                    doc_metadata["total_pages"] = len(pdf_reader)

                    for page_number, page in enumerate(pdf_reader):
                        text_page = page.get_textpage()
                        text_from_page = "\n".join(
                            text_page.get_text_range().splitlines()
                        )  # Replace \r\n
                        text_page.close()
                        image_from_page = self._extract_images_from_page(page)
                        all_text = _merge_text_and_extras(
                            [image_from_page], text_from_page
                        ).strip()
                        page.close()

                        if self.mode == "page":
                            # For legacy compatibility, add the last '\n'
                            if not all_text.endswith("\n"):
                                all_text += "\n"
                            yield Document(
                                page_content=all_text,
                                metadata={
                                    **doc_metadata,
                                    "page": page_number,
                                },
                            )
                        else:
                            full_content.append(all_text)

                    if self.mode == "single":
                        yield Document(
                            page_content=self.pages_delimitor.join(full_content),
                            metadata=doc_metadata,
                        )
                finally:
                    if pdf_reader:
                        pdf_reader.close()

    def _extract_images_from_page(self, page: "pypdfium2._helpers.page.PdfPage") -> str:
        """Extract images from a PDF page and get the text using RapidOCR.

        Args:
            page: The page object from which to extract images.

        Returns:
            str: The extracted text from the images on the page.
        """
        if not self.extract_images:
            return ""

        import pypdfium2.raw as pdfium_c

        images = list(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)))

        numpy_images = [x.get_bitmap().to_numpy() for x in images]
        for image in images:
            image.close()
        return _format_image_str.format(
            image_text=_join_images.join(self.convert_image_to_text(numpy_images))
        )


class PDFPlumberParser(ImagesPdfParser):
    """Parse a blob from a PDF using `pdfplumber` library.

    This class provides methods to parse a blob from a PDF document, supporting various
    configurations such as handling password-protected PDFs, extracting images, and
    defining extraction mode.
    It integrates the 'pdfplumber' library for PDF processing and offers synchronous
    blob parsing.

    Examples:
        Setup:

        .. code-block:: bash

            pip install -U langchain-core langchain-community pdfplumber

        Load a blob from a PDF file:

        .. code-block:: python

            from langchain_core.documents.base import Blob

            blob = Blob.from_path("./example_data/layout-parser-paper.pdf")

        Instantiate the parser:

        .. code-block:: python

            from langchain_community.document_loaders.parsers import PDFPlumberParser

            parser = PDFPlumberParser(
                # text_kwargs=None,
                # dedupe=False,
                # extract_images=False,
                # password=None,
                mode="page",
                pages_delimitor="\n\f",
                # images_to_text=None,
                # extract_tables=None,
                # extract_tables_settings=None,
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

    def __init__(
        self,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        dedupe: bool = False,
        extract_images: bool = False,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "page",
        pages_delimitor: str = _default_page_delimitor,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        extract_tables: Optional[Literal["csv", "markdown", "html"]] = None,
        extract_tables_settings: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe:  Avoiding the error of duplicate characters if `dedupe=True`
            extract_images: Whether to extract images from the PDF
            password: Optional password for opening encrypted PDFs.
            mode: The extraction mode, either "single" for extracting the entire
                document as one chunk or "page" for page-wise extraction.
            images_to_text: Optional function or callable to convert images to text
                during extraction.
            pages_delimitor: A string delimiter to separate pages in single-mode
                extraction.
            extract_tables: Whether to extract images from the PDF in a specific
                format, such as "csv", "markdown", or "html".
            extract_tables_settings: Optional dictionary of settings for customizing
            table extraction.

        Returns:
            This method does not directly return data. Use the `parse` or `lazy_parse`
            methods to retrieve parsed documents with content and metadata.

        Raises:
            ImportError: If the `pdfplumber` package is not installed.
            ValueError: If the `mode` is not "single" or "page".
            ValueError: If the `extract_tables` is not "csv", "markdown" or "html".

        """
        super().__init__(extract_images, images_to_text)
        self.password = password
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        if extract_tables and extract_tables not in ["csv", "markdown", "html"]:
            raise ValueError("mode must be csv, markdown or html")
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.dedupe = dedupe
        self.text_kwargs = text_kwargs or {}
        self.extract_tables = extract_tables
        self.extract_tables_settings = extract_tables_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_y_tolerance": 5,
            "intersection_x_tolerance": 15,
        }

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob.

        Args:
            blob: The blob to parse.

        Yields:
            An iterator over the parsed documents.
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber package not found, please install it "
                "with `pip install pdfplumber`"
            )

        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            doc = pdfplumber.open(file_path, password=self.password)  # open document
            from pdfplumber.utils import geometry  # import WordExctractor, TextMap

            contents = []
            doc_metadata = purge_metadata(
                (
                    doc.metadata
                    | {
                        "source": blob.source,
                        "file_path": blob.source,
                        "total_pages": len(doc.pages),
                    }
                )
            )
            for page in doc.pages:
                tables_bbox: list[tuple[float, float, float, float]] = (
                    self._extract_tables_bbox_from_page(page)
                )
                tables_content = self._extract_tables_from_page(page)
                images_bbox = [geometry.obj_to_bbox(image) for image in page.images]
                image_from_page = self._extract_images_from_page(page)
                page_text = []
                for content in self._split_page_content(
                    page,
                    tables_bbox,
                    tables_content,
                    images_bbox,
                    image_from_page,
                ):
                    if isinstance(content, str):  # Text
                        page_text.append(content)
                    elif isinstance(content, list):  # Table
                        page_text.append(_join_tables + self._convert_table(content))
                    else:  # Image
                        page_text.append(
                            _join_images + next(self.convert_image_to_text([content]))
                        )

                all_text = "".join(page_text).strip()

                if self.mode == "page":
                    # For legacy compatibility, add the last '\n'_
                    if not all_text.endswith("\n"):
                        all_text += "\n"
                    yield Document(
                        page_content=all_text,
                        metadata=(
                            doc_metadata
                            | {
                                "page": page.page_number - 1,
                            }
                        ),
                    )
                else:
                    contents.append(all_text)
                # PPR: add the tables_as_html and  images in all scenario ?
                # "tables_as_html": [self._convert_table_to_html(table)
                #                    for
                #                    table in tables_content],
                # "images": images_content,
                # tables_as_html.extend([self._convert_table(table)
                #                        for
                #                        table in tables_content])
            if self.mode == "single":
                yield Document(
                    page_content=self.pages_delimitor.join(contents),
                    metadata=doc_metadata,
                )

    def _process_page_content(self, page: "pdfplumber.page.Page") -> str:
        """Process the page content based on dedupe.

        Args:
            page: The PDF page to process.

        Returns:
            The extracted text from the page.
        """
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        return page.extract_text(**self.text_kwargs)

    def _split_page_content(
        self,
        page: "pdfplumber.page.Page",
        tables_bbox: list[tuple[float, float, float, float]],
        tables_content: list[list[list[Any]]],
        images_bbox: list[tuple[float, float, float, float]],
        images_content: list[np.ndarray],
        **kwargs: Any,
    ) -> Iterator[Union[str, list[list[str]], np.ndarray]]:
        """Split the page content into text, tables, and images.

        Args:
            page: The PDF page to process.
            tables_bbox: Bounding boxes of tables on the page.
            tables_content: Content of tables on the page.
            images_bbox: Bounding boxes of images on the page.
            images_content: Content of images on the page.
            **kwargs: Additional keyword arguments.

        Yields:
            An iterator over the split content (text, tables, images).
        """
        from pdfplumber.utils import (
            geometry,
            text,
        )

        # Iterate over words. If a word is in a table,
        # yield the accumulated text, and the table
        # A the word is in a previously see table, ignore it
        # Finish with the accumulated text
        kwargs.update(
            {
                "keep_blank_chars": True,
                # "use_text_flow": True,
                "presorted": True,
                "layout_bbox": kwargs.get("layout_bbox")
                # or geometry.objects_to_bbox(page.chars),
                or page.cropbox,
            }
        )
        chars = page.dedupe_chars().objects["char"] if self.dedupe else page.chars

        extractor = text.WordExtractor(
            **{k: kwargs[k] for k in text.WORD_EXTRACTOR_KWARGS if k in kwargs}
        )
        wordmap = extractor.extract_wordmap(chars)
        extract_wordmaps: list[Any] = []
        used_arrays = [False] * len(tables_bbox)
        for word, o in wordmap.tuples:
            # print(f"  Try with '{word['text']}' ...")
            is_table = False
            word_bbox = geometry.obj_to_bbox(word)
            for i, table_bbox in enumerate(tables_bbox):
                if geometry.get_bbox_overlap(word_bbox, table_bbox):
                    # Find a world in a table
                    # print("  Find in an array")
                    is_table = True
                    if not used_arrays[i]:
                        # First time I see a word in this array
                        # Yield the previous part
                        if extract_wordmaps:
                            new_wordmap = text.WordMap(tuples=extract_wordmaps)
                            new_textmap = new_wordmap.to_textmap(
                                **{
                                    k: kwargs[k]
                                    for k in text.TEXTMAP_KWARGS
                                    if k in kwargs
                                }
                            )
                            # print(f"yield {new_textmap.to_string()}")
                            yield new_textmap.to_string()
                            extract_wordmaps.clear()
                        # and yield the table
                        used_arrays[i] = True
                        # print(f"yield table {i}")
                        yield tables_content[i]
                    break
            if not is_table:
                # print(f'  Add {word["text"]}')
                extract_wordmaps.append((word, o))
        if extract_wordmaps:
            # Text after the array ?
            new_wordmap = text.WordMap(tuples=extract_wordmaps)
            new_textmap = new_wordmap.to_textmap(
                **{k: kwargs[k] for k in text.TEXTMAP_KWARGS if k in kwargs}
            )
            # print(f"yield {new_textmap.to_string()}")
            yield new_textmap.to_string()
        # Add images-
        for content in images_content:
            yield content

    def _extract_images_from_page(
        self, page: "pdfplumber.page.Page"
    ) -> list[np.ndarray]:
        """Extract images from a PDF page.

        Args:
            page: The PDF page to extract images from.

        Returns:
            A list of extracted images as numpy arrays.
        """
        from PIL import Image

        if not self.extract_images:
            return []

        images = []
        for img in page.images:
            if img["stream"]["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    np.frombuffer(img["stream"].get_data(), dtype=np.uint8).reshape(
                        img["stream"]["Height"], img["stream"]["Width"], -1
                    )
                )
            elif img["stream"]["Filter"].name in _PDF_FILTER_WITH_LOSS:
                buf = np.frombuffer(img["stream"].get_data(), dtype=np.uint8)
                images.append(np.array(Image.open(io.BytesIO(buf.tobytes()))))
            else:
                logger.warning("Unknown PDF Filter!")

        return images

    def _extract_tables_bbox_from_page(
        self,
        page: "pdfplumber.page.Page",
    ) -> list[tuple]:
        """Extract bounding boxes of tables from a PDF page.

        Args:
            page: The PDF page to extract table bounding boxes from.

        Returns:
            A list of bounding boxes for tables on the page.
        """
        if not self.extract_tables:
            return []
        from pdfplumber.table import TableSettings

        table_settings = self.extract_tables_settings
        tset = TableSettings.resolve(table_settings)
        return [table.bbox for table in page.find_tables(tset)]

    def _extract_tables_from_page(
        self,
        page: "pdfplumber.page.Page",
    ) -> list[list[list[Any]]]:
        """Extract tables from a PDF page.

        Args:
            page: The PDF page to extract tables from.

        Returns:
            A list of tables, where each table is a list of rows, and each row is a
            list of cell values.
        """
        if not self.extract_tables:
            return []
        table_settings = self.extract_tables_settings
        tables_list = page.extract_tables(table_settings)
        return tables_list

    def _convert_table(self, table: list[list[str]]) -> str:
        """Convert a table to the specified format.

        Args:
            table: The table to convert.

        Returns:
            The table content as a string in the specified format.
        """
        format = self.extract_tables
        if format is None:
            return ""
        if format == "markdown":
            return self._convert_table_to_markdown(table)
        elif format == "html":
            return self._convert_table_to_html(table)
        elif format == "csv":
            return self._convert_table_to_csv(table)
        else:
            raise ValueError(f"Unknown table format: {format}")

    def _convert_table_to_csv(self, table: list[list[str]]) -> str:
        """Convert a table to CSV format.

        Args:
            table: The table to convert.

        Returns:
            The table content as a string in CSV format.
        """
        if not table:
            return ""

        output = ["\n\n"]

        # skip first row in details if header is part of the table
        # j = 0 if self.header.external else 1

        # iterate over detail rows
        for row in table:
            line = ""
            for i, cell in enumerate(row):
                # output None cells with empty string
                cell = "" if cell is None else cell.replace("\n", " ")
                line += cell + ","
            output.append(line)
        return "\n".join(output) + "\n\n"

    def _convert_table_to_html(self, table: list[list[str]]) -> str:
        """
        Convert table content as a string in HTML format.
        If clean is true, markdown syntax is removed from cell content.

        Args:
            table: The table to convert.

        Returns:
            The table content as a string in HTML format.
        """
        if not len(table):
            return ""
        output = "<table>\n"
        clean = True

        # iterate over detail rows
        for row in table:
            line = "<tr>"
            for i, cell in enumerate(row):
                # output None cells with empty string
                cell = "" if cell is None else cell.replace("\n", " ")
                if clean:  # remove sensitive syntax
                    cell = html.escape(cell.replace("-", "&#45;"))
                line += "<td>" + cell + "</td>"
            line += "</tr>\n"
            output += line
        return output + "</table>\n"

    def _convert_table_to_markdown(self, table: list[list[str]]) -> str:
        """Convert table content as a string in Github-markdown format.

        Args:
            table: The table to convert.

        Returns:
            The table content as a string in Markdown format.
        """
        clean = False
        if not table:
            return ""
        col_count = len(table[0])

        output = "|" + "|".join("" for i in range(col_count)) + "|\n"
        output += "|" + "|".join("---" for i in range(col_count)) + "|\n"

        # skip first row in details if header is part of the table
        # j = 0 if self.header.external else 1

        # iterate over detail rows
        for row in table:
            line = "|"
            for i, cell in enumerate(row):
                # output None cells with empty string
                cell = "" if cell is None else cell.replace("\n", " ")
                if clean:  # remove sensitive syntax
                    cell = html.escape(cell.replace("-", "&#45;"))
                line += cell + "|"
            line += "\n"
            output += line
        return output + "\n"


# %% --------- Online pdf loader ---------
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
        linearization_config: Optional["TextLinearizationConfig"] = None,
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
                metadata={"source": blob.source, "page": idx + 1},
                # type: ignore[attr-defined]
            )


class DocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Form Recognizer) and chunks at character level."""

    def __init__(self, client: Any, model: str):
        logger.warning(
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
