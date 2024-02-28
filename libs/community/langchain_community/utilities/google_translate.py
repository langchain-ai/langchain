from typing import Any, Dict, List, Optional, Union

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator


class GoogleTranslateApiWarper(BaseModel):
    """Wrapper for Google Translate API.

    Free and does not require any setup.
    """

    service_urls: List[str] = ["translate.google.com"]
    src: str = "auto"
    dest: str = "en"
    proxy: Any = None
    timeout: Optional[Union[int, float]] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_language_codes(cls, values: dict) -> dict:
        """Validate that src and dest language codes are valid."""
        src_language_codes = [
            "auto",
            "am",
            "ar",
            "eu",
            "bn",
            "en-GB",
            "pt-BR",
            "bg",
            "ca",
            "chr",
            "hr",
            "cs",
            "da",
            "nl",
            "en",
            "et",
            "fil",
            "fi",
            "fr",
            "de",
            "el",
            "gu",
            "iw",
            "hi",
            "hu",
            "is",
            "id",
            "it",
            "ja",
            "kn",
            "ko",
            "lv",
            "lt",
            "ms",
            "ml",
            "mr",
            "no",
            "pl",
            "pt-PT",
            "ro",
            "ru",
            "sr",
            "zh-CN",
            "sk",
            "sl",
            "es",
            "sw",
            "sv",
            "ta",
            "te",
            "th",
            "zh-TW",
            "tr",
            "ur",
            "uk",
            "vi",
            "cy",
        ]
        dest_language_codes = [
            "am",
            "ar",
            "eu",
            "bn",
            "en-GB",
            "pt-BR",
            "bg",
            "ca",
            "chr",
            "hr",
            "cs",
            "da",
            "nl",
            "en",
            "et",
            "fil",
            "fi",
            "fr",
            "de",
            "el",
            "gu",
            "iw",
            "hi",
            "hu",
            "is",
            "id",
            "it",
            "ja",
            "kn",
            "ko",
            "lv",
            "lt",
            "ms",
            "ml",
            "mr",
            "no",
            "pl",
            "pt-PT",
            "ro",
            "ru",
            "sr",
            "zh-CN",
            "sk",
            "sl",
            "es",
            "sw",
            "sv",
            "ta",
            "te",
            "th",
            "zh-TW",
            "tr",
            "ur",
            "uk",
            "vi",
            "cy",
        ]

        if values["src"] not in src_language_codes:
            raise ValueError(f"Invalid src language code: {values['src']}")
        if values["dest"] not in dest_language_codes:
            raise ValueError(f"Invalid dest language code: {values['dest']}")

        try:
            from googletrans import Translator  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import googletrans python package. "
                "Please install it with `pip install -U googletrans==4.0.0-rc1`."
            )

        return values

    def _translate(self, text: str) -> Dict[str, str]:
        """Translate text from src to dest language."""
        from googletrans import Translator

        translator = Translator(service_urls=self.service_urls)
        result = translator.translate(text, src=self.src, dest=self.dest)

        return {
            "src": result.src,
            "dest": result.dest,
            "text": result.text,
            "origin": result.origin,
            "pronunciation": result.pronunciation,
        }

    def _detect(self, text: str) -> Dict[str, Union[str, float]]:
        """Detect language of text."""
        from googletrans import Translator

        translator = Translator(service_urls=self.service_urls)
        result = translator.detect(text)

        return {
            "lang": result.lang,
            "confidence": result.confidence,
        }

    def run(self, query: str) -> str:
        """Run query through Google Translate and return translated text."""
        return self._translate(query).get("text")
