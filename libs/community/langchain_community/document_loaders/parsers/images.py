import base64
import html
import io
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Iterator, Literal

import numpy
import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)


class BaseImageBlobParser(BaseBlobParser):
    """
    Abstract base class for parsing image blobs into text.

    Attributes:
        format (Literal["text", "markdown", "html"]):
          Output format of the parsed text.
    """

    def __init__(
        self,
        *,
        format: Literal["text", "markdown", "html"] = "text",
    ):
        """
        Initializes the BaseImageBlobParser.

        Args:
            format (Literal["text", "markdown", "html"]):
              The format for the parsed output.
        """
        self.format = format

    @abstractmethod
    def _analyze_image(self, img: "Image") -> str:
        """
        Abstract method to analyze an image and extract textual content.

        Args:
            img (Image):
              The image to be analyzed.

        Returns:
            str:
              The extracted text content.
        """
        pass

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Lazily parses a blob and yields Document objects containing the parsed content.

        Args:
            blob (Blob):
              The blob to be parsed.

        Yields:
            Document:
              A document containing the parsed content and metadata.
        """
        try:
            from PIL import Image as Img

            with blob.as_bytes_io() as buf:
                if blob.mimetype == "application/x-npy":
                    img = Img.fromarray(numpy.load(buf))
                else:
                    img = Img.open(buf)
                content = self._analyze_image(img)
                if content:
                    source = blob.source or "#"
                    if self.format == "markdown":
                        content = content.replace("]", r"\\]")
                        content = f"![{content}]({source})"
                    elif self.format == "html":
                        content = (
                            f'<img alt="{html.escape(content, quote=True)} '
                            f'src="{source}" />'
                        )
                logger.debug("Image text: %s", content.replace("\n", "\\n"))
                yield Document(
                    page_content=content,
                    metadata={**blob.metadata, **{"source": blob.source}},
                )
        except ImportError:
            raise ImportError(
                "`Pillow` package not found, please install it with "
                "`pip install Pillow`"
            )


class RapidOCRBlobParser(BaseImageBlobParser):
    """
    Parser for extracting text from images using the RapidOCR library.

    Attributes:
        ocr:
          The RapidOCR instance for performing OCR.
    """

    def __init__(
        self,
        *,
        format: Literal["text", "markdown", "html"] = "text",
    ):
        """
        Initializes the RapidOCRBlobParser.

        Args:
            format (Literal["text", "markdown", "html"]):
              The format for the parsed output.
        """
        super().__init__(format=format)
        self.ocr = None

    def _analyze_image(self, img: "Image") -> str:
        """
        Analyzes an image and extracts text using RapidOCR.

        Args:
            img (Image):
              The image to be analyzed.

        Returns:
            str:
              The extracted text content.
        """
        if not self.ocr:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self.ocr = RapidOCR()
            except ImportError:
                raise ImportError(
                    "`rapidocr-onnxruntime` package not found, please install it with "
                    "`pip install rapidocr-onnxruntime`"
                )
        ocr_result, _ = self.ocr(np.array(img))  # type: ignore
        content = ""
        if ocr_result:
            content = ("\n".join([text[1] for text in ocr_result])).strip()
        return content


class TesseractBlobParser(BaseImageBlobParser):
    """
    Parser for extracting text from images using the Tesseract OCR library.

    Attributes:
        langs (list[str]):
          The languages to use for OCR.
    """

    def __init__(
        self,
        *,
        format: Literal["text", "markdown", "html"] = "text",
        langs: Iterable[str] = ("eng",),
    ):
        """
        Initializes the TesseractBlobParser.

        Args:
            format (Literal["text", "markdown", "html"]):
              The format for the parsed output.
            langs (list[str]):
              The languages to use for OCR.
        """
        super().__init__(format=format)
        self.langs = list(langs)

    def _analyze_image(self, img: "Image") -> str:
        """
        Analyzes an image and extracts text using Tesseract OCR.

        Args:
            img (Image):
              The image to be analyzed.

        Returns:
            str: The extracted text content.
        """
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "`pytesseract` package not found, please install it with "
                "`pip install pytesseract`"
            )
        return pytesseract.image_to_string(img, lang="+".join(self.langs)).strip()


_PROMPT_IMAGES_TO_DESCRIPTION = (
    "You are an assistant tasked with summarizing "
    "images for retrieval. "
    "These summaries will be embedded and used to retrieve the raw image. "
    "Give a concise summary of the image that is well optimized for retrieval "
    "and extract all the text from the image."
)


class LLMImageBlobParser(BaseImageBlobParser):
    """
    Parser for analyzing images using a language model (LLM).

    Attributes:
        model (BaseChatModel):
          The language model to use for analysis.
        prompt (str):
          The prompt to provide to the language model.
    """

    def __init__(
        self,
        *,
        format: Literal["text", "markdown", "html"] = "text",
        model: BaseChatModel,
        prompt: str = _PROMPT_IMAGES_TO_DESCRIPTION,
    ):
        """
        Initializes the LLMImageBlobParser.

        Args:
            format (Literal["text", "markdown", "html"]):
              The format for the parsed output.
            model (BaseChatModel):
              The language model to use for analysis.
            prompt (str):
              The prompt to provide to the language model.
        """
        super().__init__(format=format)
        self.model = model
        self.prompt = prompt

    def _analyze_image(self, img: "Image") -> str:
        """
        Analyzes an image using the provided language model.

        Args:
            img (Image):
              The image to be analyzed.

        Returns:
            str: *
              The extracted textual content.
        """
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        img_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        msg = self.model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": self.prompt},
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
        return result
