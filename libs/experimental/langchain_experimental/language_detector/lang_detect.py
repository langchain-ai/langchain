from typing import Any, List, Tuple

from langchain_experimental.language_detector.base import LanguageDetectorBase


def _import_langdetect() -> Any:
    try:
        import langdetect
    except ImportError as e:
        raise ImportError(
            "Could not import langdetect, please install with "
            "`pip install langdetect`."
        ) from e
    return langdetect


class LangDetector(LanguageDetectorBase):
    """
    Language detector based on langdetect package.
    Warning: Results are not deterministic!
    It supports 55 languages out of the box.
    """

    def _detect_single(self, text: str) -> str:
        """Detects the most probable language of a single text."""
        langdetect = _import_langdetect()
        return langdetect.detect(text)

    def _detect_many(
        self, text: str, threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """Detects all languages of a single text with a score bigger than threshold.
        Returns them sorted on score in descending order.
        """
        langdetect = _import_langdetect()
        predictions = langdetect.detect_langs(text)
        return [(x.lang, x.prob) for x in predictions if x.prob > threshold]
