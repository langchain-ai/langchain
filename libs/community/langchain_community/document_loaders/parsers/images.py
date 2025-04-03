import base64
import io
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Iterator

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
    """Abstract base class for parsing image blobs into text."""

    @abstractmethod
    def _analyze_image(self, img: "Image") -> str:
        """Abstract method to analyze an image and extract textual content.

        Args:
            img: The image to be analyzed.

        Returns:
          The extracted text content.
        """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse a blob and yields Documents containing the parsed content.

        Args:
            blob (Blob): The blob to be parsed.

        Yields:
            Document:
              A document containing the parsed content and metadata.
        """
        try:
            from PIL import Image as Img
        except ImportError:
            raise ImportError(
                "`Pillow` package not found, please install it with "
                "`pip install Pillow`"
            )

        with blob.as_bytes_io() as buf:
            if blob.mimetype == "application/x-npy":
                array = numpy.load(buf)
                if array.ndim == 3 and array.shape[2] == 1:  # Grayscale image
                    img = Img.fromarray(numpy.squeeze(array, axis=2), mode="L")
                else:
                    img = Img.fromarray(array)
            else:
                img = Img.open(buf)
            content = self._analyze_image(img)
            logger.debug("Image text: %s", content.replace("\n", "\\n"))
            yield Document(
                page_content=content,
                metadata={**blob.metadata, **{"source": blob.source}},
            )


class RapidOCRBlobParser(BaseImageBlobParser):
    """Parser for extracting text from images using the RapidOCR library.

    Attributes:
        ocr:
          The RapidOCR instance for performing OCR.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes the RapidOCRBlobParser.
        """
        super().__init__()
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
    """Parse for extracting text from images using the Tesseract OCR library."""

    def __init__(
        self,
        *,
        langs: Iterable[str] = ("eng",),
    ):
        """Initialize the TesseractBlobParser.

        Args:
            langs (list[str]):
              The languages to use for OCR.
        """
        super().__init__()
        self.langs = list(langs)

    def _analyze_image(self, img: "Image") -> str:
        """Analyze an image and extracts text using Tesseract OCR.

        Args:
            img: The image to be analyzed.

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


_PROMPT_IMAGES_TO_DESCRIPTION: str = (
    "You are an assistant tasked with summarizing images for retrieval. "
    "1. These summaries will be embedded and used to retrieve the raw image. "
    "Give a concise summary of the image that is well optimized for retrieval\n"
    "2. extract all the text from the image. "
    "Do not exclude any content from the page.\n"
    "Format answer in markdown without explanatory text "
    "and without markdown delimiter ``` at the beginning. "
)


class LLMImageBlobParser(BaseImageBlobParser):
    """Parser for analyzing images using a language model (LLM).

    Attributes:
        model (BaseChatModel):
          The language model to use for analysis.
        prompt (str):
          The prompt to provide to the language model.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel,
        prompt: str = _PROMPT_IMAGES_TO_DESCRIPTION,
    ):
        """Initializes the LLMImageBlobParser.

        Args:
            model (BaseChatModel):
              The language model to use for analysis.
            prompt (str):
              The prompt to provide to the language model.
        """
        super().__init__()
        self.model = model
        self.prompt = prompt

    def _analyze_image(self, img: "Image") -> str:
        """Analyze an image using the provided language model.

        Args:
            img: The image to be analyzed.

        Returns:
            The extracted textual content.
        """
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        img_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        msg = self.model.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": self.prompt.format(format=format),
                        },
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
