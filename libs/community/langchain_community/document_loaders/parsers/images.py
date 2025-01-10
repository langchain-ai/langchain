import base64
import html
import io
import logging
from abc import abstractmethod

from PIL import Image
from typing import Iterator, Literal

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class ImageBlobParser(BaseBlobParser):
    def __init__(
            self,
            *,
            format: Literal["text", "markdown", "html"] = "text",
    ):
        self.format = format

    @abstractmethod
    def _analyze_image(self, img: Image) -> str:
        pass

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as buf:
            img = Image.open(buf)
            content = self._analyze_image(img)
            if content:
                if self.format == "markdown":
                    content = content.replace("]", r"\\]")
                    content = f"![{content}](.)"
                elif self.format == "html":
                    content = f'<img alt="{html.escape(content, quote=True)}" />'
            logger.debug("Image text: %s", content.replace("\n", "\\n"))
            yield Document(
                page_content=content,
                metadata={"source": blob.source},
            )


class RapidOCRBlobParser(ImageBlobParser):
    def __init__(
            self,
            *,
            format: Literal["text", "markdown", "html"] = "text",
    ):
        super().__init__(format=format)
        self.ocr = None

    def _analyze_image(self, img: Image) -> str:
        if not self.ocr:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                raise ImportError(
                    "`rapidocr-onnxruntime` package not found, please install it with "
                    "`pip install rapidocr-onnxruntime`"
                )
            self.ocr = RapidOCR()
        ocr_result, _ = self.ocr(img)
        content = ""
        if ocr_result:
            content = ("\n".join([text[1] for text in ocr_result])).strip()
        return content


class TesseractBlobParser(ImageBlobParser):

    def __init__(
            self,
            *,
            format: Literal["text", "markdown", "html"] = "text",
            langs: list[str] = ["eng"],

    ):
        super().__init__(format=format)
        self.langs = langs

    def _analyze_image(self, img: Image) -> str:
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "`pytesseract` package not found, please install it with "
                "`pip install pytesseract`"
            )
        return pytesseract.image_to_string(img, lang="+".join(self.langs)).strip()


_prompt_images_to_description = (
    "You are an assistant tasked with summarizing "
    "images for retrieval. "
    "These summaries will be embedded and used to retrieve the raw image. "
    "Give a concise summary of the image that is well optimized for retrieval "
    "and extract all the text from the image.")


class MultimodalBlobParser(ImageBlobParser):

    def __init__(
            self,
            *,
            format: Literal["text", "markdown", "html"] = "text",
            model: BaseChatModel,
            prompt: str = _prompt_images_to_description,

    ):
        super().__init__(format=format)
        self.model = model
        self.prompt = prompt

    def _analyze_image(self, img: Image) -> str:
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
